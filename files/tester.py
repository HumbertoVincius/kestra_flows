import os
import json
import subprocess
import tempfile
import shutil
import base64
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union, List, Dict

from dotenv import load_dotenv
from supabase import Client, create_client
from llm_client import call_llm as llm_call_llm, openai_client, anthropic_client, gemini_client

try:
    from github import Github, GithubException, Auth
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("⚠️  PyGithub não encontrado. Instale com: pip install PyGithub")

# === Caminhos e configuração ===
ENV_PATH = Path(__file__).parent.parent / ".env"
CONFIG_PATH = Path(__file__).parent.parent / "system" / "tester_config.json"

load_dotenv(dotenv_path=ENV_PATH)


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️  Falha ao ler tester_config.json: {exc}")
        return {}


def _normalize_str(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _serialize_payload(payload: Union[str, dict, list, None]) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, dict):
        if not payload:
            return None
        if "content" in payload and isinstance(payload["content"], str):
            content = payload["content"].strip()
            if content:
                return content
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(payload)
    if isinstance(payload, Iterable) and not isinstance(payload, (bytes, bytearray)):
        payload = list(payload)
        if not payload:
            return None
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return "\n".join(str(item) for item in payload)
    return str(payload)


def _extract_config_values(config: dict) -> Tuple[
    dict,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str]
]:
    parameters = config.get("parameters") or {}
    system_payload = config.get("system_message")
    user_payload = config.get("user_message")

    ai_model = _normalize_str(parameters.get("ai_model"))
    provider = _normalize_str(parameters.get("provider"))

    normalized_system = _serialize_payload(system_payload)
    normalized_user = _serialize_payload(user_payload)

    return parameters, normalized_system, normalized_user, ai_model, provider


TESTER_CONFIG = _load_config()
(
    CONFIG_PARAMETERS,
    CONFIG_SYSTEM_MESSAGE,
    CONFIG_USER_MESSAGE,
    CONFIG_AI_MODEL,
    CONFIG_PROVIDER,
) = _extract_config_values(TESTER_CONFIG)

CONFIG_PROVIDER = CONFIG_PROVIDER.lower() if CONFIG_PROVIDER else None
CONFIG_TEMPERATURE = CONFIG_PARAMETERS.get("temperature")
CONFIG_MAX_TOKENS = CONFIG_PARAMETERS.get("max_tokens")
CONFIG_TOP_P = CONFIG_PARAMETERS.get("top_p")
CONFIG_FREQUENCY_PENALTY = CONFIG_PARAMETERS.get("frequency_penalty")
CONFIG_PRESENCE_PENALTY = CONFIG_PARAMETERS.get("presence_penalty")
CONFIG_STOP = CONFIG_PARAMETERS.get("stop")

# === Clientes LLM ===
# Clientes importados de llm_client.py

# === Cliente Supabase ===
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    if ENV_PATH.exists():
        env_content = ENV_PATH.read_text(encoding="utf-8-sig")
        for line in env_content.strip().split("\n"):
            if line.startswith("SUPABASE_URL=") and not supabase_url:
                supabase_url = line.split("=", 1)[1].strip()
            elif line.startswith("SUPABASE_KEY=") and not supabase_key:
                supabase_key = line.split("=", 1)[1].strip()
            elif line.startswith("SUPABASE_SERVICE_ROLE_KEY=") and not supabase_service_key:
                supabase_service_key = line.split("=", 1)[1].strip()

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e/ou SUPABASE_KEY não encontradas no arquivo .env")

supabase: Client = create_client(supabase_url, supabase_key)
supabase_write: Optional[Client] = None

# === Constantes de agente ===
PROJECT_ID = "639e810b-9d8c-4f31-9569-ecf61fb43888"
TESTER_AGENT_NAME = "tester_agent"
CODEGEN_AGENT_NAME = "codegen_agent"
DEPLOY_AGENT_NAME = "deploy_agent"
MESSAGE_CONTENT_CREATED = "test_report_created"


def get_schema_summary(schema_id: str) -> str:
    if not schema_id:
        raise ValueError("schema_id inválido para buscar schema")

    try:
        response = (
            supabase.table("schema_documents")
            .select("content")
            .eq("schema_id", schema_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar schema com schema_id={schema_id}: {exc}") from exc

    if not response.data:
        raise ValueError(f"Schema com schema_id={schema_id} não encontrado em schema_documents")

    record = response.data[0]
    content = record.get("content") or {}
    schema_summary = content.get("schema_summary") if isinstance(content, dict) else content

    try:
        return json.dumps(schema_summary, ensure_ascii=False, indent=2)
    except Exception:
        return str(schema_summary)


# === Supabase helpers ===
def get_system_message() -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    try:
        response = (
            supabase.table("system_message")
            .select("content, system_revision, ai_id, updated_at")
            .eq("is_active", True)
            .eq("agent_type", "tester_agent")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='tester_agent'")
        record = response.data[0]
        content = record.get("content")
        revision = record.get("system_revision")
        ai_id = record.get("ai_id")
        ai_model = None
        ai_provider = None
        if not content:
            raise ValueError("Campo 'content' está vazio no registro encontrado")
        if ai_id:
            ai_response = (
                supabase.table("ai_models")
                .select("ai_model, provider")
                .eq("ai_id", ai_id)
                .limit(1)
                .execute()
            )
            if ai_response.data:
                ai_model = ai_response.data[0].get("ai_model")
                ai_provider = ai_response.data[0].get("provider")
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, dict):
            content_str = json.dumps(content, ensure_ascii=False)
        else:
            content_str = str(content)
        return content_str, revision, ai_model, ai_provider
    except Exception as exc:
        print(f"Erro ao buscar system message do Supabase: {exc}")
        raise


def get_codegen_from_message() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], List[Dict[str, str]], Optional[List[str]]]:
    """
    Busca a mensagem mais recente de codegen_agent -> tester_agent em agent_messages
    e retorna o codegen correspondente, com prd_id, scaffold_id e artifacts.
    """
    try:
        response = (
            supabase.table("agent_messages")
            .select("id, codegen_id, prd_id, scaffold_id, schema_id, status, message_content")
            .eq("project_id", PROJECT_ID)
            .eq("from_agent", CODEGEN_AGENT_NAME)
            .eq("to_agent", TESTER_AGENT_NAME)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar agent_messages para tester_agent: {exc}") from exc

    if not response.data:
        # Nenhuma mensagem pendente para este agente
        return None, None, None, None, None, None, [], None

    msg_record = response.data[0]
    message_id = msg_record.get("id")
    codegen_id = msg_record.get("codegen_id")
    prd_id = msg_record.get("prd_id")
    scaffold_id = msg_record.get("scaffold_id")
    schema_id = msg_record.get("schema_id")
    message_content = msg_record.get("message_content")

    if not message_id or not codegen_id:
        raise ValueError(
            "Mensagem encontrada em agent_messages não contém id ou codegen_id válido. "
            "Verifique se o codegen_agent está salvando codegen_id corretamente."
        )

    # Buscar codegen correspondente em codegen_documents
    try:
        codegen_response = (
            supabase.table("codegen_documents")
            .select("codegen_id, prd_id, scaffold_id, content")
            .eq("codegen_id", codegen_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar Codegen com codegen_id={codegen_id}: {exc}") from exc

    if not codegen_response.data:
        raise ValueError(
            f"Codegen com codegen_id={codegen_id} não encontrado em codegen_documents"
        )

    record = codegen_response.data[0]
    # Se prd_id ou scaffold_id não vieram da mensagem, usar os do registro
    prd_id = prd_id or record.get("prd_id")
    scaffold_id = scaffold_id or record.get("scaffold_id")
    content = record.get("content") or {}

    # Extrair artifacts
    artifacts: List[Dict[str, str]] = []
    if isinstance(content, dict):
        content_data = content.get("content") if isinstance(content.get("content"), dict) else content
        artifacts_list = content_data.get("artifacts", []) if isinstance(content_data, dict) else []
        if isinstance(artifacts_list, list):
            artifacts = [
                {"path": a.get("path", ""), "content": a.get("content", "")}
                for a in artifacts_list
                if isinstance(a, dict) and a.get("path")
            ]

    # Extrair corrected_files do message_content se for JSON estruturado
    corrected_files = None
    if message_content:
        try:
            if isinstance(message_content, str):
                content_parsed = json.loads(message_content)
            else:
                content_parsed = message_content
            
            if isinstance(content_parsed, dict) and content_parsed.get("type") == "test_request":
                corrected_files = content_parsed.get("corrected_files", [])
                if corrected_files:
                    print(f"📋 Arquivos corrigidos identificados no message_content: {len(corrected_files)} arquivos")
        except (json.JSONDecodeError, TypeError):
            # message_content não é JSON, é string simples (criação inicial)
            pass

    # Formatar content para texto
    content_text = json.dumps(content, ensure_ascii=False, indent=2)
    return content_text, codegen_id, prd_id, scaffold_id, schema_id, message_id, artifacts, corrected_files


def extract_artifacts_from_codegen(codegen_content: str) -> List[Dict[str, str]]:
    """Extrai lista de artifacts do content do codegen."""
    try:
        if isinstance(codegen_content, str):
            content_dict = json.loads(codegen_content)
        else:
            content_dict = codegen_content
        
        artifacts = []
        content_data = content_dict.get("content") if isinstance(content_dict, dict) else content_dict
        if isinstance(content_data, dict):
            artifacts_list = content_data.get("artifacts", [])
            if isinstance(artifacts_list, list):
                artifacts = [
                    {"path": a.get("path", ""), "content": str(a.get("content", ""))}
                    for a in artifacts_list
                    if isinstance(a, dict) and a.get("path")
                ]
        return artifacts
    except Exception as exc:
        print(f"⚠️  Erro ao extrair artifacts: {exc}")
        return []


def run_eslint(artifacts: List[Dict[str, str]], temp_dir: Path) -> str:
    """Executa ESLint nos arquivos gerados e retorna output."""
    if not artifacts:
        return "Nenhum artifact para validar com ESLint"
    
    try:
        # Criar estrutura de diretórios e arquivos
        for artifact in artifacts:
            file_path = artifact.get("path", "")
            file_content = artifact.get("content", "")
            if not file_path:
                continue
            
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(file_content, encoding="utf-8")
        
        # Verificar se package.json existe, se não, criar um básico
        package_json_path = temp_dir / "package.json"
        if not package_json_path.exists():
            package_json = {
                "name": "temp-validation",
                "version": "1.0.0",
                "scripts": {
                    "lint": "eslint . --ext .ts,.tsx,.js,.jsx"
                }
            }
            package_json_path.write_text(json.dumps(package_json, indent=2), encoding="utf-8")
        
        # Tentar executar ESLint
        try:
            result = subprocess.run(
                ["npx", "eslint", ".", "--ext", ".ts,.tsx,.js,.jsx", "--format", "json"],
                cwd=str(temp_dir),
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout + result.stderr if result.stdout or result.stderr else "ESLint executado sem erros"
        except subprocess.TimeoutExpired:
            return "ESLint timeout após 60 segundos"
        except FileNotFoundError:
            return "ESLint não encontrado. Instale com: npm install -g eslint"
        except Exception as e:
            return f"Erro ao executar ESLint: {str(e)}"
    except Exception as e:
        return f"Erro ao preparar arquivos para ESLint: {str(e)}"


def run_build(artifacts: List[Dict[str, str]], temp_dir: Path) -> str:
    """Tenta executar build do Next.js e retorna output."""
    if not artifacts:
        return "Nenhum artifact para fazer build"
    
    try:
        # Criar estrutura de diretórios e arquivos
        for artifact in artifacts:
            file_path = artifact.get("path", "")
            file_content = artifact.get("content", "")
            if not file_path:
                continue
            
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(file_content, encoding="utf-8")
        
        # Verificar/criar package.json básico para Next.js
        package_json_path = temp_dir / "package.json"
        if not package_json_path.exists():
            package_json = {
                "name": "temp-validation",
                "version": "1.0.0",
                "scripts": {
                    "build": "next build"
                },
                "dependencies": {
                    "next": "^14.0.0",
                    "react": "^18.0.0",
                    "react-dom": "^18.0.0"
                }
            }
            package_json_path.write_text(json.dumps(package_json, indent=2), encoding="utf-8")
        
        # Tentar executar build (sem instalar dependências, apenas verificar estrutura)
        try:
            # Primeiro tentar verificar se Next.js está disponível
            result = subprocess.run(
                ["npx", "next", "build", "--dry-run"],
                cwd=str(temp_dir),
                capture_output=True,
                text=True,
                timeout=120
            )
            output = result.stdout + result.stderr
            if not output or "command not found" in output.lower():
                # Tentar build normal
                result = subprocess.run(
                    ["npm", "run", "build"],
                    cwd=str(temp_dir),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                output = result.stdout + result.stderr
            return output if output else "Build executado sem output"
        except subprocess.TimeoutExpired:
            return "Build timeout após 300 segundos"
        except FileNotFoundError:
            return "npm/npx não encontrado. Instale Node.js e npm"
        except Exception as e:
            return f"Erro ao executar build: {str(e)}"
    except Exception as e:
        return f"Erro ao preparar arquivos para build: {str(e)}"


def build_user_message(artifacts: List[Dict[str, str]], eslint_output: str, build_output: str, schema_text: Optional[str], base_prompt: str) -> str:
    """Constrói mensagem do usuário para LLM com artifacts, schema e outputs de validação."""
    sections = []
    
    if base_prompt:
        sections.append(base_prompt.strip())
    
    if schema_text:
        sections.append("[SCHEMA]")
        sections.append(schema_text.strip())

    sections.append("[ARTIFACTS]")
    sections.append(f"Lista de {len(artifacts)} arquivos gerados para validação:")
    for idx, artifact in enumerate(artifacts, 1):
        file_path = artifact.get("path", "")
        file_content = artifact.get("content", "")
        sections.append(f"\n--- Arquivo {idx}: {file_path} ---")
        sections.append(file_content)
    
    if eslint_output:
        sections.append("\n[ESLINT_OUTPUT]")
        sections.append(eslint_output)
    
    if build_output:
        sections.append("\n[BUILD_OUTPUT]")
        sections.append(build_output)
    
    return "\n".join(sections)


def call_llm(
    system_message: Optional[str],
    user_message: str,
    model: Optional[str],
    provider: Optional[str],
    system_revision: Optional[str] = None,
    max_tokens: int = 12000,
) -> dict:
    """Chama LLM para gerar relatório de validação."""
    if isinstance(model, str):
        model = model.strip() or None
    if isinstance(provider, str):
        provider = provider.strip().lower() or None

    need_lookup = system_message is None or model is None or provider is None

    fetched_message = fetched_revision = fetched_model = fetched_provider = None
    if need_lookup:
        fetched_message, fetched_revision, fetched_model, fetched_provider = get_system_message()
        if system_message is None:
            system_message = fetched_message
        if system_revision is None:
            system_revision = fetched_revision
        if model is None:
            model = fetched_model
        if provider is None:
            provider = fetched_provider

    if system_message is None:
        raise ValueError("System message não disponível para o Tester Agent")

    if model is None:
        model = CONFIG_AI_MODEL or "gpt-4o"

    if provider is None:
        if isinstance(model, str) and model.lower().startswith("claude"):
            provider = "anthropic"
        else:
            provider = CONFIG_PROVIDER or "openai"
    provider = provider.lower()

    if provider == "openai" and isinstance(model, str) and model.lower().startswith("claude"):
        print("⚠️  Modelo claude solicitado com provider openai; usando gpt-4o por padrão.")
        model = "gpt-4o"

    try:
        result = llm_call_llm(
            system_message=system_message,
            user_message=user_message,
            model=model,
            provider=provider,
            system_revision=system_revision,
            max_tokens=max_tokens,
            default_max_tokens=12000,
            default_temperature=0.1,
            agent_name="Tester Agent",
            get_system_message_fn=get_system_message,
            config_ai_model=CONFIG_AI_MODEL,
            config_provider=CONFIG_PROVIDER,
            config_max_tokens=CONFIG_MAX_TOKENS,
            config_temperature=CONFIG_TEMPERATURE,
            config_top_p=CONFIG_TOP_P,
            config_frequency_penalty=CONFIG_FREQUENCY_PENALTY,
            config_presence_penalty=CONFIG_PRESENCE_PENALTY,
            config_stop=CONFIG_STOP,
        )
        
        raw_output = result["raw_output"]
        usage_info = result["metadata"]

        if raw_output is None:
            raise ValueError("Resposta da LLM vazia")

        # Parse do relatório
        parsed_report = parse_test_report(raw_output)

        # Metadados
        metadata = {
            "prompt_tokens": usage_info["prompt_tokens"],
            "completion_tokens": usage_info["completion_tokens"],
            "total_tokens": usage_info["total_tokens"],
            "agent_model": model,
            "provider": provider,
            "system_revision": system_revision or "",
        }

        return {
            "content": parsed_report,
            "metadata": metadata,
            "raw_output": raw_output
        }

    except Exception as e:
        print(f"Erro ao chamar a LLM: {e}")
        raise


def parse_test_report(raw: Any) -> dict:
    """Parse do relatório de validação retornado pela LLM."""
    if isinstance(raw, dict):
        parsed = raw
    else:
        if raw is None:
            raise ValueError("Resposta da LLM vazia para o Tester Agent")
        raw_str = str(raw).strip()
        if not raw_str:
            raise ValueError("Resposta da LLM vazia para o Tester Agent")

        # Tentar extrair JSON
        raw_str = _extract_code_fence(raw_str)
        parsed = _parse_jsonish(raw_str)
        if parsed is None:
            raise ValueError(
                "Não foi possível converter a resposta da LLM em JSON válido. Trecho inicial: "
                + raw_str[:200]
            )

    # Normalizar estrutura do relatório
    if not isinstance(parsed, dict):
        raise ValueError("Resposta da LLM não é um objeto JSON válido")

    # Garantir estrutura esperada
    report = parsed.get("report", {})
    if not isinstance(report, dict):
        report = {}

    summary = report.get("summary", {})
    files = report.get("files", [])
    
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(files, list):
        files = []

    # Garantir campos obrigatórios no summary
    summary.setdefault("overall_status", "not_ok")
    summary.setdefault("total_files", len(files))
    summary.setdefault("files_ok", 0)
    summary.setdefault("files_with_errors", 0)
    summary.setdefault("summary", "")

    # Normalizar arquivos
    normalized_files = []
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        normalized_file = {
            "file_path": file_item.get("file_path", ""),
            "status": file_item.get("status", "error"),
            "syntax_errors": file_item.get("syntax_errors", []),
            "logic_errors": file_item.get("logic_errors", []),
            "build_errors": file_item.get("build_errors", []),
            "suggestions": file_item.get("suggestions", [])
        }
        normalized_files.append(normalized_file)

    return {
        "report": {
            "summary": summary,
            "files": normalized_files
        },
        "eslint_output": parsed.get("eslint_output", ""),
        "build_output": parsed.get("build_output", "")
    }


def _extract_code_fence(raw_str: str) -> str:
    """Extrai conteúdo de code fence se presente."""
    import re
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_str, re.IGNORECASE)
    return match.group(1).strip() if match else raw_str


def _parse_jsonish(raw_str: str) -> Any:
    """Tenta parsear string como JSON de várias formas."""
    import ast
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(raw_str)
    except (ValueError, SyntaxError):
        pass

    try:
        normalized = raw_str.replace("'", '"')
        return json.loads(normalized)
    except json.JSONDecodeError:
        pass

    # Tentar extrair primeiro objeto JSON
    start = raw_str.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(raw_str)):
            char = raw_str[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw_str[start : idx + 1])
                        except json.JSONDecodeError:
                            pass

    return None


def ensure_github_repo(token: str, owner: str, repo_name: str):
    """
    Verifica se o repositório GitHub existe. Se não existir, cria.
    
    Args:
        token: GitHub personal access token
        owner: Usuário ou organização que possui o repositório
        repo_name: Nome do repositório
    
    Returns:
        Objeto do repositório GitHub
    
    Raises:
        ValueError: Se token ou owner não forem fornecidos
        GithubException: Se houver erro na API do GitHub
    """
    if not GITHUB_AVAILABLE:
        raise ValueError("PyGithub não está disponível. Instale com: pip install PyGithub")
    
    if not token or not token.strip():
        raise ValueError("GITHUB_TOKEN não configurado")
    
    if not owner or not owner.strip():
        raise ValueError("GITHUB_OWNER não configurado")
    
    try:
        g = Github(auth=Auth.Token(token))
        user = g.get_user()
        
        # Verificar se owner é o próprio usuário autenticado ou uma organização
        try:
            if owner == user.login:
                org_or_user = user
            else:
                org_or_user = g.get_organization(owner)
        except GithubException:
            # Se não conseguir encontrar como organização, assume que é o usuário
            org_or_user = user
        
        # Tentar obter o repositório
        try:
            repo = org_or_user.get_repo(repo_name)
            print(f"✅ Repositório '{repo_name}' já existe")
            return repo
        except GithubException as e:
            if e.status == 404:
                # Repositório não existe, criar
                print(f"📦 Criando repositório '{repo_name}'...")
                if owner == user.login:
                    repo = user.create_repo(
                        repo_name,
                        private=True,  # Pode ser ajustado conforme necessário
                        auto_init=False,  # Não criar README inicial
                        description="Projeto gerado automaticamente pelo codegen agent"
                    )
                else:
                    repo = org_or_user.create_repo(
                        repo_name,
                        private=True,
                        auto_init=False,
                        description="Projeto gerado automaticamente pelo codegen agent"
                    )
                print(f"✅ Repositório '{repo_name}' criado com sucesso")
                return repo
            else:
                raise
    
    except GithubException as e:
        error_msg = f"Erro ao verificar/criar repositório GitHub: {e}"
        if e.status == 401:
            error_msg += " (Token inválido ou sem permissões)"
        elif e.status == 403:
            error_msg += " (Sem permissão para criar repositórios)"
        print(f"❌ {error_msg}")
        raise


def push_codegen_to_github(repo, artifacts: List[Dict[str, str]], prd_id: Optional[str], scaffold_id: Optional[str], codegen_id: Optional[str], tester_id: Optional[str] = None) -> bool:
    """
    Faz commit e push dos artifacts gerados para o repositório GitHub.
    
    Args:
        repo: Objeto do repositório GitHub
        artifacts: Lista de artifacts com 'path' e 'content'
        prd_id: ID do PRD relacionado
        scaffold_id: ID do scaffold relacionado
        codegen_id: ID do codegen gerado
        tester_id: ID do tester que validou (opcional)
    
    Returns:
        True se sucesso, False caso contrário
    """
    if not GITHUB_AVAILABLE:
        print("⚠️  PyGithub não está disponível. Pulando push para GitHub.")
        return False
    
    if not artifacts or len(artifacts) == 0:
        print("⚠️  Nenhum artifact para fazer push")
        return False
    
    try:
        # Determinar branch padrão (main ou master)
        default_branch = repo.default_branch
        branch_name = default_branch if default_branch else "main"
        
        # Verificar se repositório tem commits (está vazio ou não)
        repo_is_empty = False
        try:
            repo.get_branch(branch_name)
        except GithubException as e:
            if e.status == 404:
                # Branch não existe, repositório está vazio
                repo_is_empty = True
                print(f"ℹ️  Repositório vazio detectado. Usando branch '{branch_name}' para primeiro commit...")
            else:
                raise
        
        # Montar mensagem de commit
        commit_message_parts = ["Codegen validado pelo Tester"]
        if tester_id:
            commit_message_parts.append(f"tester_id={tester_id}")
        if codegen_id:
            commit_message_parts.append(f"codegen_id={codegen_id}")
        if prd_id:
            commit_message_parts.append(f"prd_id={prd_id}")
        if scaffold_id:
            commit_message_parts.append(f"scaffold_id={scaffold_id}")
        commit_message = ": ".join(commit_message_parts)
        
        # Processar cada artifact
        files_created = 0
        files_updated = 0
        errors = []
        
        for artifact in artifacts:
            file_path = artifact.get("path")
            file_content = artifact.get("content", "")
            
            if not file_path or not file_path.strip():
                errors.append(f"Artifact sem path válido: {artifact}")
                continue
            
            file_path = file_path.strip()
            
            # Validar conteúdo (deve ser string)
            if not isinstance(file_content, str):
                try:
                    file_content = str(file_content)
                except Exception as e:
                    errors.append(f"Erro ao converter conteúdo de {file_path}: {e}")
                    continue
            
            # PyGithub espera o conteúdo decodificado (texto simples), não base64
            # A biblioteca faz a codificação internamente quando envia para a API
            # Se o conteúdo vier codificado em base64, precisamos decodificar primeiro
            content_to_upload = file_content
            
            # Tentar detectar e decodificar se estiver em base64
            # Base64 geralmente tem apenas letras, números, +, /, = e pode ter quebras de linha
            if file_content and len(file_content.strip()) > 10:
                stripped = file_content.strip().replace('\n', '').replace('\r', '').replace(' ', '')
                # Heurística: se o conteúdo parece base64 (só tem caracteres base64 válidos)
                # e é uma string sem espaços ou caracteres especiais comuns em código
                if len(stripped) > 20 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in stripped):
                    # Pode ser base64, tentar decodificar
                    try:
                        decoded_content = base64.b64decode(stripped).decode('utf-8')
                        if decoded_content and len(decoded_content) > 0:
                            # Decodificou com sucesso, usar o conteúdo decodificado
                            content_to_upload = decoded_content
                    except Exception:
                        # Se falhou na decodificação, não é base64 ou está corrompido
                        # Usar conteúdo original
                        pass
            
            # Se repositório está vazio, todos os arquivos são novos (não precisa verificar)
            if repo_is_empty:
                try:
                    # Se é o primeiro arquivo do repositório vazio, não especificar branch
                    # O GitHub criará a branch automaticamente com o primeiro commit
                    if files_created == 0:
                        # Primeiro arquivo - não especificar branch para criar a branch
                        repo.create_file(
                            file_path,
                            commit_message,
                            content_to_upload
                        )
                        # Após criar o primeiro arquivo, atualizar flag para próximos
                        repo_is_empty = False
                        branch_name = repo.default_branch or "main"
                    else:
                        # Arquivos subsequentes - usar branch (já criada no primeiro commit)
                        repo.create_file(
                            file_path,
                            commit_message,
                            content_to_upload,
                            branch=branch_name
                        )
                    files_created += 1
                except Exception as e:
                    errors.append(f"Erro ao criar {file_path}: {str(e)}")
            else:
                # Repositório não está vazio, verificar se arquivo existe
                try:
                    existing_file = repo.get_contents(file_path, ref=branch_name)
                    # Arquivo existe, atualizar
                    try:
                        repo.update_file(
                            file_path,
                            commit_message,
                            content_to_upload,
                            existing_file.sha,
                            branch=branch_name
                        )
                        files_updated += 1
                    except Exception as e:
                        errors.append(f"Erro ao atualizar {file_path}: {str(e)}")
                except GithubException as e:
                    if e.status == 404:
                        # Arquivo não existe, criar
                        try:
                            repo.create_file(
                                file_path,
                                commit_message,
                                content_to_upload,
                                branch=branch_name
                            )
                            files_created += 1
                        except Exception as e:
                            errors.append(f"Erro ao criar {file_path}: {str(e)}")
                    else:
                        errors.append(f"Erro ao verificar {file_path}: {str(e)}")
        
        # Resumo
        if errors:
            print(f"⚠️  Push concluído com erros:")
            for error in errors[:5]:  # Mostrar até 5 erros
                print(f"   - {error}")
            if len(errors) > 5:
                print(f"   ... e mais {len(errors) - 5} erros")
        
        print(f"✅ GitHub: {files_created} arquivos criados, {files_updated} arquivos atualizados")
        return len(errors) == 0
    
    except GithubException as e:
        error_msg = f"Erro ao fazer push para GitHub: {e}"
        if e.status == 401:
            error_msg += " (Token inválido ou expirado)"
        elif e.status == 403:
            error_msg += " (Sem permissão para escrever no repositório)"
        elif e.status == 404:
            error_msg += " (Repositório não encontrado)"
        elif e.status == 422:
            error_msg += f" (Erro de validação: {e.data if hasattr(e, 'data') else 'dados inválidos'})"
        print(f"❌ {error_msg}")
        if hasattr(e, 'data'):
            print(f"   Detalhes: {e.data}")
        return False
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Erro inesperado ao fazer push para GitHub: {e}")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Detalhes: {error_details}")
        return False


def save_to_tester_documents(
    result: dict,
    codegen_id: Optional[str],
    prd_id: Optional[str],
    scaffold_id: Optional[str],
    schema_id: Optional[str],
    eslint_output: str,
    build_output: str
) -> dict:
    """Salva relatório de teste na tabela tester_documents."""
    if not codegen_id:
        raise ValueError("codegen_id inválido: é necessário para registrar tester_documents")

    content_jsonb = {
        "metadata": result.get("metadata"),
        "report": result.get("content", {}).get("report", {}),
        "eslint_output": eslint_output,
        "build_output": build_output,
        "raw_output": result.get("raw_output"),
    }

    is_using_service = supabase_service_key is not None and supabase_service_key.strip() != ""
    write_client = create_client(supabase_url, supabase_service_key) if is_using_service else supabase

    insert_payload = {
        "project_id": PROJECT_ID,
        "codegen_id": codegen_id,
        "prd_id": prd_id,
        "scaffold_id": scaffold_id,
        "schema_id": schema_id,
        "content": content_jsonb,
    }

    response = write_client.table("tester_documents").insert(insert_payload).execute()

    if not response.data:
        raise ValueError("Nenhum registro foi inserido na tabela tester_documents")

    tester_record = response.data[0]
    tester_id = tester_record.get("tester_id")

    # Extrair arquivos com erro do relatório para message_content
    report = result.get("content", {}).get("report", {})
    files_with_errors_list = []
    if isinstance(report, dict):
        report_files = report.get("files", [])
        if isinstance(report_files, list):
            for file_info in report_files:
                if isinstance(file_info, dict):
                    file_path = file_info.get("file_path", "")
                    file_status = file_info.get("status", "")
                    if file_path and file_status == "error":
                        files_with_errors_list.append(file_path)
    
    # Determinar destinatário e message_content baseado na presença de erros
    if files_with_errors_list:
        # Se houver erros: enviar para codegen_agent para correção
        to_agent = CODEGEN_AGENT_NAME
        message_content = json.dumps({
            "type": "correction_request",
            "files_with_errors": files_with_errors_list
        }, ensure_ascii=False)
    else:
        # Se não houver erros: enviar para deploy_agent para deploy
        to_agent = DEPLOY_AGENT_NAME
        message_content = MESSAGE_CONTENT_CREATED

    try:
        write_client.table("agent_messages")\
            .insert({
                "project_id": PROJECT_ID,
                "from_agent": TESTER_AGENT_NAME,
                "to_agent": to_agent,
                "status": "pending",
                "message_content": message_content,
                "prd_id": prd_id,
                "scaffold_id": scaffold_id,
                "schema_id": schema_id,
                "codegen_id": codegen_id,
                "tester_id": tester_id,
            })\
            .execute()
        if files_with_errors_list:
            print(f"📤 Mensagem enviada para {to_agent} com {len(files_with_errors_list)} arquivos com erro (correção necessária)")
        else:
            print(f"📤 Mensagem enviada para {to_agent} (código validado, pronto para deploy)")
    except Exception as log_error:
        print(f"⚠️  Falha ao registrar mensagem em agent_messages: {log_error}")

    return tester_record


if __name__ == "__main__":
    # Carregar configuração
    user_msg = CONFIG_USER_MESSAGE
    system_msg = CONFIG_SYSTEM_MESSAGE
    ai_model = CONFIG_AI_MODEL
    provider = CONFIG_PROVIDER
    system_rev: Optional[str] = None
    message_id: Optional[str] = None

    if not user_msg:
        raise ValueError(
            "Defina user_message em system/tester_config.json ou forneça user_message explicitamente."
        )

    try:
        # Fallback para Supabase quando informações essenciais não estão no arquivo local
        if system_msg is None or ai_model is None or provider is None:
            fetched_msg, fetched_rev, fetched_model, fetched_provider = get_system_message()
            if system_msg is None:
                system_msg = fetched_msg
            system_rev = fetched_rev
            if ai_model is None:
                ai_model = fetched_model
            if provider is None:
                provider = fetched_provider

        if system_msg is None:
            raise ValueError("System message não encontrado. Configure em tester_config.json ou no Supabase.")

        if ai_model is None:
            ai_model = "gpt-4o"

        if provider is None:
            provider = "openai"

        print("Parâmetros efetivos da execução:")
        print(f" - model: {ai_model}")
        print(f" - provider: {provider}")
        print(f" - notes: {CONFIG_PARAMETERS.get('notes') or '<não informado>'}")

        max_tokens_value = CONFIG_MAX_TOKENS if isinstance(CONFIG_MAX_TOKENS, int) else 12000

        # Buscar codegen a partir da mensagem pendente mais recente enviada pelo codegen_agent
        print("\n📥 Buscando codegen a partir de agent_messages...")
        codegen_content, codegen_id, prd_id, scaffold_id, schema_id, message_id, artifacts, corrected_files = get_codegen_from_message()

        if not codegen_id or not artifacts or not message_id:
            print("no pending messages")
            raise SystemExit(0)

        # Filtrar artifacts se houver lista de arquivos corrigidos
        original_artifact_count = len(artifacts)
        if corrected_files and len(corrected_files) > 0:
            artifacts = [a for a in artifacts if a.get("path", "").strip() in corrected_files]
            print(f"🧪 Testando apenas arquivos corrigidos: {len(artifacts)} arquivos (de {original_artifact_count} total)")
        else:
            print(f"🧪 Testando todos os arquivos (criação inicial): {len(artifacts)} arquivos")

        print(f"✅ Codegen encontrado: codegen_id={codegen_id}")
        print(f"   - prd_id: {prd_id}")
        print(f"   - scaffold_id: {scaffold_id}")
        print(f"   - artifacts para teste: {len(artifacts)} arquivos")

        # Marcar mensagem como em processamento
        try:
            result = supabase.table("agent_messages")\
                .update({"status": "working"})\
                .eq("id", message_id)\
                .eq("project_id", PROJECT_ID)\
                .execute()
            if result.data:
                print(f"✅ Mensagem {message_id} marcada como 'working'")
            else:
                print(f"⚠️  Nenhuma linha atualizada ao marcar mensagem {message_id} como 'working'")
        except Exception as exc:
            print(f"❌ Falha ao marcar mensagem como working: {exc}")
            raise

        # Executar validações
        print("\n🔍 Executando validações...")
        eslint_output = ""
        build_output = ""

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            print("  - Executando ESLint...")
            eslint_output = run_eslint(artifacts, temp_dir)
            print(f"    ESLint concluído (output: {len(eslint_output)} chars)")

            print("  - Executando build...")
            build_output = run_build(artifacts, temp_dir)
            print(f"    Build concluído (output: {len(build_output)} chars)")

        # Construir mensagem do usuário (incluindo schema, se disponível)
        schema_text = get_schema_summary(schema_id) if schema_id else ""
        user_message = build_user_message(artifacts, eslint_output, build_output, schema_text, user_msg)

        # Chamar LLM para gerar relatório
        print("\n🤖 Gerando relatório de validação...")
        resultado = call_llm(
            system_message=system_msg,
            user_message=user_message,
            model=ai_model,
            provider=provider,
            system_revision=system_rev,
            max_tokens=max_tokens_value,
        )

        llm_meta = resultado["metadata"]
        total_tokens = llm_meta["total_tokens"]
        print(f"✅ Relatório gerado: {total_tokens} total tokens")

        # Extrair resumo do relatório
        report = resultado.get("content", {}).get("report", {})
        summary = report.get("summary", {})
        overall_status = summary.get("overall_status", "not_ok")
        total_files = summary.get("total_files", 0)
        files_ok = summary.get("files_ok", 0)
        files_with_errors = summary.get("files_with_errors", 0)

        print(f"\n📊 Resumo da validação:")
        print(f"   - Status geral: {overall_status}")
        print(f"   - Total de arquivos: {total_files}")
        print(f"   - Arquivos OK: {files_ok}")
        print(f"   - Arquivos com erros: {files_with_errors}")

        # Salvar no banco
        print("\n💾 Salvando relatório no Supabase...")
        saved_record = save_to_tester_documents(
            resultado,
            codegen_id,
            prd_id,
            scaffold_id,
            schema_id,
            eslint_output,
            build_output
        )

        tester_id = saved_record.get("tester_id")
        print(f"✅ Relatório salvo com sucesso: tester_id={tester_id}")

        # Integração GitHub - apenas se não houver erros
        if overall_status == "ok" and files_with_errors == 0:
            github_token = os.getenv("GITHUB_TOKEN")
            github_owner = os.getenv("GITHUB_OWNER")
            repo_name = "projeto_we_build"

            if github_token and github_owner:
                try:
                    print(f"\n🔗 Integrando com GitHub (owner: {github_owner}, repo: {repo_name})...")
                    
                    # Verificar/criar repositório
                    repo = ensure_github_repo(github_token, github_owner, repo_name)
                    
                    if artifacts:
                        # Fazer push dos arquivos
                        push_success = push_codegen_to_github(
                            repo,
                            artifacts,
                            prd_id,
                            scaffold_id,
                            codegen_id,
                            tester_id
                        )
                        if push_success:
                            print(f"✅ Código validado e enviado para GitHub: https://github.com/{github_owner}/{repo_name}")
                        else:
                            print("⚠️  Push para GitHub concluído com erros (verifique logs acima)")
                    else:
                        print("⚠️  Nenhum artifact encontrado para fazer push")
                
                except ValueError as e:
                    print(f"⚠️  GitHub não configurado corretamente: {e}")
                except Exception as e:
                    print(f"⚠️  Erro ao integrar com GitHub: {e}")
                    print("   Continuando execução normalmente...")
            else:
                if not github_token:
                    print("ℹ️  GITHUB_TOKEN não configurado. Pulando integração GitHub.")
                if not github_owner:
                    print("ℹ️  GITHUB_OWNER não configurado. Pulando integração GitHub.")
        else:
            print(f"\n⚠️  Código não enviado para GitHub devido a erros encontrados (status: {overall_status}, arquivos com erros: {files_with_errors})")

        # Marcar mensagem original como done
        try:
            result = supabase.table("agent_messages")\
                .update({"status": "done"})\
                .eq("id", message_id)\
                .eq("project_id", PROJECT_ID)\
                .execute()
            if result.data:
                print(f"✅ Mensagem {message_id} marcada como 'done'")
            else:
                print(f"⚠️  Nenhuma linha atualizada ao marcar mensagem {message_id} como 'done'")
        except Exception as exc:
            print(f"❌ Falha ao marcar mensagem como done: {exc}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Erro na execução do tester_agent: {e}")
        if message_id:
            try:
                supabase.table("agent_messages")\
                    .update({
                        "status": "error",
                        "message_content": f"tester_error: {str(e)}",
                    })\
                    .eq("id", message_id)\
                    .eq("project_id", PROJECT_ID)\
                    .execute()
            except Exception as exc:
                print(f"⚠️  Falha ao marcar mensagem como error: {exc}")
        raise

