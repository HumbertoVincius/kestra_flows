import os
import json
import ast
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client

# === Carregamento de configuração ===
ENV_PATH = Path(__file__).parent.parent / ".env"
CONFIG_PATH = Path(__file__).parent.parent / "system" / "scaffold_config.json"

load_dotenv(dotenv_path=ENV_PATH)


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️  Falha ao ler scaffold_config.json: {exc}")
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


SCAFFOLD_CONFIG = _load_config()
(
    CONFIG_PARAMETERS,
    CONFIG_SYSTEM_MESSAGE,
    CONFIG_USER_MESSAGE,
    CONFIG_AI_MODEL,
    CONFIG_PROVIDER,
) = _extract_config_values(SCAFFOLD_CONFIG)

CONFIG_PROVIDER = CONFIG_PROVIDER.lower() if CONFIG_PROVIDER else None
CONFIG_TEMPERATURE = CONFIG_PARAMETERS.get("temperature")
CONFIG_MAX_TOKENS = CONFIG_PARAMETERS.get("max_tokens")
CONFIG_TOP_P = CONFIG_PARAMETERS.get("top_p")
CONFIG_FREQUENCY_PENALTY = CONFIG_PARAMETERS.get("frequency_penalty")
CONFIG_PRESENCE_PENALTY = CONFIG_PARAMETERS.get("presence_penalty")
CONFIG_STOP = CONFIG_PARAMETERS.get("stop")

# === Clientes LLM ===
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    if ENV_PATH.exists():
        env_content = ENV_PATH.read_text(encoding="utf-8-sig")
        for line in env_content.strip().split("\n"):
            if line.startswith("OPENAI_API_KEY="):
                openai_api_key = line.split("=", 1)[1].strip()
                break
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")

openai_client = OpenAI(api_key=openai_api_key)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_api_key:
    try:
        from anthropic import Anthropic  # type: ignore

        anthropic_client = Anthropic(api_key=anthropic_api_key)
    except ImportError:  # pragma: no cover
        anthropic_client = None
        print("⚠️  Biblioteca 'anthropic' não encontrada; defina provider como openai ou instale o pacote.")
else:
    anthropic_client = None

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
SCAFFOLD_AGENT_NAME = "scaffold_agent"
CODEGEN_AGENT_NAME = "codegen_agent"
MESSAGE_CONTENT_CREATED = "scaffold_created"

PRD_AGENT_NAME = "prd_agent"
SCHEMA_AGENT_NAME = "schema_agent"


# === Utilitários de parsing ===
def _extract_code_fence(raw_str: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_str, re.IGNORECASE)
    return match.group(1).strip() if match else raw_str


def _extract_first_json_object(raw_str: str) -> Optional[str]:
    """Extrai o primeiro objeto JSON completo de uma string."""
    start = raw_str.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(raw_str)):
        char = raw_str[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw_str[start : idx + 1]
    return None


def _extract_json_array(raw_str: str) -> Optional[str]:
    """
    Extrai um array JSON completo, mesmo se truncado.
    Retorna o array mais completo possível ou None se não encontrar.
    """
    start = raw_str.find("[")
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape = False
    
    for idx in range(start, len(raw_str)):
        char = raw_str[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return raw_str[start : idx + 1]
    
    # Se não encontrou fechamento completo, retorna None (não queremos array parcial)
    return None


def _extract_all_json_objects(raw_str: str) -> list:
    """
    Extrai todos os objetos JSON válidos de uma string.
    Útil para validar se conseguimos extrair tudo que está no raw_output.
    """
    objects = []
    start = 0
    
    while start < len(raw_str):
        obj_start = raw_str.find("{", start)
        if obj_start == -1:
            break
        
        obj_str = _extract_first_json_object(raw_str[obj_start:])
        if obj_str:
            try:
                obj = json.loads(obj_str)
                objects.append(obj)
                start = obj_start + len(obj_str)
            except json.JSONDecodeError:
                start = obj_start + 1
        else:
            start = obj_start + 1
    
    return objects


def _count_artifacts_in_raw(raw_str: str) -> int:
    """
    Conta quantos artifacts (objetos com 'path') existem no raw_output.
    Usado para validar se extraímos tudo.
    """
    # Contar ocorrências de "path" que parecem ser chaves de artifacts
    # Procura por padrão: "path": ou "path":
    path_pattern = r'"path"\s*:'
    matches = re.findall(path_pattern, raw_str)
    return len(matches)


def _parse_jsonish(raw_str: str) -> Any:
    """
    Versão melhorada do parser que tenta múltiplas estratégias.
    Prioriza parsing completo sobre parcial.
    """
    # Estratégia 1: Tentar parsear como JSON completo (preferido)
    try:
        parsed = json.loads(raw_str)
        return parsed
    except json.JSONDecodeError:
        pass
    
    # Estratégia 2: Tentar extrair array JSON completo
    array_str = _extract_json_array(raw_str)
    if array_str:
        try:
            parsed = json.loads(array_str)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Estratégia 3: Tentar parsear como Python literal
    try:
        parsed = ast.literal_eval(raw_str)
        return parsed
    except (ValueError, SyntaxError):
        pass
    
    # Estratégia 4: Tentar normalizar aspas simples para duplas
    try:
        normalized = raw_str.replace("'", '"')
        parsed = json.loads(normalized)
        return parsed
    except json.JSONDecodeError:
        pass
    
    # NÃO usar fallback de primeiro objeto - queremos tudo ou nada
    return None


ARTIFACT_KEYS = [
    "files_root",
    "files_app",
    "files_lib",
    "files_api",
    "files_test",
]


def _distribute_artifacts_to_groups(artifacts: list) -> dict:
    """
    Distribui artifacts em grupos baseado no path quando a LLM retornar apenas 'artifacts'.
    Esta função é usada como fallback quando a LLM não respeita a estrutura de grupos.
    """
    groups = {key: [] for key in ARTIFACT_KEYS}
    
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        
        path = artifact.get("path", "")
        if not isinstance(path, str):
            continue
        
        path_lower = path.lower()
        
        # Distribuir baseado no path
        if path_lower.startswith("app/api/"):
            groups["files_api"].append(artifact)
        elif path_lower.startswith("app/"):
            groups["files_app"].append(artifact)
        elif path_lower.startswith("src/lib/") or path_lower.startswith("src/styles/"):
            # src/lib/ e src/styles/ vão para files_lib
            groups["files_lib"].append(artifact)
        elif any(path_lower.endswith(ext) for ext in [".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"]):
            groups["files_test"].append(artifact)
        else:
            # Arquivos na raiz ou outros
            groups["files_root"].append(artifact)
    
    return groups


def _guess_code_language(path: str) -> str:
    extension = Path(path).suffix.lower()
    mapping = {
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".json": "json",
        ".py": "python",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cs": "csharp",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".md": "markdown",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".sql": "sql",
        ".sh": "bash",
        ".xml": "xml",
        ".c": "c",
        ".cpp": "cpp",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".nix": "nix",
    }
    return mapping.get(extension, "")


def _format_prd_payload(prd_payload: Any, raw_output: Optional[str]) -> str:
    sections: list[str] = []

    if isinstance(prd_payload, dict):
        summary = {key: value for key, value in prd_payload.items() if key != "artifacts"}
        artifacts = prd_payload.get("artifacts")

        if summary:
            try:
                summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                summary_text = str(summary)
            sections.append("[PRD SUMMARY]")
            sections.append(summary_text)

        if isinstance(artifacts, list) and artifacts:
            artifact_lines: list[str] = ["[PRD ARTIFACTS]"]
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                path = str(artifact.get("path", "") or "").strip() or "<sem-path>"
                content = artifact.get("content")
                if isinstance(content, (dict, list)):
                    content_text = json.dumps(content, ensure_ascii=False, indent=2)
                elif content is None:
                    content_text = ""
                else:
                    content_text = str(content)

                language = _guess_code_language(path)
                fence_header = f"```{language}" if language else "```"

                artifact_lines.append(f"--- {path} ---")
                artifact_lines.append(fence_header)
                artifact_lines.append(content_text)
                artifact_lines.append("```")
                artifact_lines.append("")

            if artifact_lines and artifact_lines[-1] == "":
                artifact_lines.pop()

            sections.append("\n".join(artifact_lines))

        if sections:
            return "\n\n".join(sections)

    if isinstance(prd_payload, (dict, list)):
        try:
            return json.dumps(prd_payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            pass

    if isinstance(raw_output, str) and raw_output.strip():
        return raw_output.strip()

    return str(prd_payload) if prd_payload is not None else ""


def _normalize_artifact_entry(entry: Any) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(f"Artifact inválido: {entry}")
    path = entry.get("path")
    content = entry.get("content")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"Artifact sem path válido: {entry}")
    if not isinstance(content, str):
        if content is None:
            content = ""
        else:
            content = json.dumps(content, ensure_ascii=False, indent=2)
    return {"path": path.strip(), "content": content}


def _normalize_scaffold_structure(data: Union[list, dict]) -> dict:
    """
    Normaliza a estrutura do scaffold preservando TODOS os artifacts.
    Se houver erro ao normalizar algum item, propaga a exceção (não queremos resultado parcial).
    Evita duplicação entre artifacts e files_root quando vierem da mesma fonte.
    Distribui automaticamente em grupos quando a LLM retornar apenas 'artifacts'.
    """
    result: dict[str, Any] = {}
    artifact_groups = {key: [] for key in ARTIFACT_KEYS}

    if isinstance(data, list):
        # Lista: distribuir em grupos baseado no path
        normalized_list = []
        for idx, item in enumerate(data):
            try:
                normalized_item = _normalize_artifact_entry(item)
                normalized_list.append(normalized_item)
            except Exception as e:
                raise ValueError(
                    f"Erro ao normalizar item {idx} da lista: {e}. "
                    f"Item: {json.dumps(item, ensure_ascii=False)[:200]}"
                ) from e
        
        if normalized_list:
            # Distribuir em grupos
            distributed = _distribute_artifacts_to_groups(normalized_list)
            for key in ARTIFACT_KEYS:
                items = distributed.get(key, [])
                artifact_groups[key] = items
                result[key] = items  # Sempre incluir, mesmo se vazio
            
            # NÃO criar artifacts - arquivos ficam apenas nos grupos
        
        return result

    if not isinstance(data, dict):
        raise ValueError("Estrutura retornada pela LLM não é lista nem objeto")

    # Verificar se há grupos específicos ou apenas 'artifacts'
    has_specific_groups = any(key in ARTIFACT_KEYS for key in data.keys())
    has_only_artifacts = "artifacts" in data and not has_specific_groups

    # Processar dict - se algum item falhar, propaga erro
    for key, value in data.items():
        if key == "artifacts" and isinstance(value, list):
            normalized_artifacts = []
            for idx, item in enumerate(value):
                try:
                    normalized_item = _normalize_artifact_entry(item)
                    normalized_artifacts.append(normalized_item)
                except Exception as e:
                    raise ValueError(
                        f"Erro ao normalizar artifact {idx} em 'artifacts': {e}. "
                        f"Item: {json.dumps(item, ensure_ascii=False)[:200]}"
                    ) from e
            
            # Se a LLM retornou apenas 'artifacts' sem grupos, distribuir automaticamente
            if has_only_artifacts:
                print("⚠️  LLM retornou apenas 'artifacts'. Distribuindo automaticamente em grupos baseado nos paths...")
                distributed = _distribute_artifacts_to_groups(normalized_artifacts)
                for group_key in ARTIFACT_KEYS:
                    items = distributed.get(group_key, [])
                    artifact_groups[group_key] = items
                    result[group_key] = items  # Sempre incluir, mesmo se vazio
                    if items:
                        print(f"  ✅ {len(items)} arquivos distribuídos em {group_key}")
                    else:
                        print(f"  ⚪ {group_key} vazio (sem arquivos neste grupo)")
                # NÃO criar artifacts - arquivos ficam apenas nos grupos
            else:
                # Se há grupos específicos junto com artifacts, ignorar artifacts e usar apenas os grupos
                # Não adicionar artifacts ao result
                pass
                
        elif key in ARTIFACT_KEYS and isinstance(value, list):
            normalized_items = []
            for idx, item in enumerate(value):
                try:
                    normalized_item = _normalize_artifact_entry(item)
                    normalized_items.append(normalized_item)
                except Exception as e:
                    raise ValueError(
                        f"Erro ao normalizar {key}[{idx}]: {e}. "
                        f"Item: {json.dumps(item, ensure_ascii=False)[:200]}"
                    ) from e
            artifact_groups[key] = normalized_items
            result[key] = normalized_items
        else:
            result[key] = value

    # Remover artifacts se existir - arquivos devem estar apenas nos grupos
    if "artifacts" in result:
        del result["artifacts"]
    
    # Garantir que TODOS os grupos estejam sempre presentes no result, mesmo que vazios
    # Isso mantém a estrutura consistente para o codegen agent
    for key in ARTIFACT_KEYS:
        if key not in result:
            # Se o grupo não existe no result, usar do artifact_groups ou lista vazia
            result[key] = artifact_groups.get(key, [])
        elif not isinstance(result.get(key), list):
            # Se existe mas não é lista, garantir que seja lista vazia
            result[key] = []

    return result


# === LLM Parsing ===
def parse_scaffold_content(raw: Any, raw_output_str: Optional[str] = None) -> dict:
    """
    Parse e valida o conteúdo do scaffold com validação rigorosa.
    
    Se não conseguir extrair TODOS os artifacts do raw_output, lança erro.
    Não retorna resultado parcial.
    
    Args:
        raw: Dados brutos da LLM (pode ser dict, list ou string)
        raw_output_str: String original do raw_output para validação (opcional)
    
    Returns:
        dict: Estrutura normalizada com todos os artifacts
        
    Raises:
        ValueError: Se não conseguir extrair todos os artifacts ou se houver perda de dados
    """
    parsed = None
    raw_str_for_validation = None
    
    if isinstance(raw, dict):
        parsed = raw
        raw_str_for_validation = json.dumps(raw, ensure_ascii=False) if raw_output_str is None else raw_output_str
    elif isinstance(raw, list):
        parsed = raw
        raw_str_for_validation = json.dumps(raw, ensure_ascii=False) if raw_output_str is None else raw_output_str
    else:
        if raw is None:
            raise ValueError("Resposta da LLM vazia para o Scaffold")
        
        raw_str = str(raw).strip()
        if not raw_str:
            raise ValueError("Resposta da LLM vazia para o Scaffold")
        
        raw_str_for_validation = raw_str
        raw_str = _extract_code_fence(raw_str)
        parsed = _parse_jsonish(raw_str)
        
        if parsed is None:
            # Tentar extrair todos os objetos para ver quantos existem
            all_objects = _extract_all_json_objects(raw_str)
            if all_objects:
                count_in_raw = len(all_objects)
                raise ValueError(
                    f"Não foi possível converter a resposta da LLM em JSON válido completo. "
                    f"Encontrados {count_in_raw} objetos JSON válidos, mas não foi possível parsear como estrutura completa. "
                    f"Isso pode indicar que a resposta foi truncada. "
                    f"Trecho inicial: {raw_str[:500]}"
                )
            raise ValueError(
                f"Não foi possível converter a resposta da LLM em JSON válido. "
                f"Trecho inicial: {raw_str[:500]}"
            )
    
    # Normalizar estrutura
    normalized = _normalize_scaffold_structure(parsed)
    
    # Validação rigorosa: contar artifacts nos grupos (não há mais campo artifacts)
    artifacts_count = 0
    for key in ARTIFACT_KEYS:
        items = normalized.get(key, [])
        if isinstance(items, list):
            artifacts_count += len(items)
    
    if artifacts_count == 0:
        raise ValueError(
            "Nenhum artifact foi extraído da resposta da LLM. "
            "A estrutura normalizada está vazia."
        )
    
    # Se temos raw_output_str, validar que extraímos tudo
    if raw_str_for_validation:
        expected_count = _count_artifacts_in_raw(raw_str_for_validation)
        
        if expected_count > 0 and artifacts_count < expected_count:
            # Tentar extrair todos os objetos para diagnóstico
            all_objects = _extract_all_json_objects(raw_str_for_validation)
            extracted_count = len(all_objects)
            
            raise ValueError(
                f"PERDA DE DADOS DETECTADA: "
                f"O raw_output contém {expected_count} artifacts (detectados por 'path'), "
                f"mas apenas {artifacts_count} foram normalizados. "
                f"Conseguimos extrair {extracted_count} objetos JSON válidos. "
                f"Isso indica que a resposta pode estar truncada ou malformada. "
                f"Não retornando resultado parcial para evitar dados inconsistentes. "
                f"Considere aumentar max_tokens ou verificar se a resposta foi completada."
            )
        
        # Log de sucesso
        if expected_count > 0:
            print(f"✅ Validação: {artifacts_count} artifacts extraídos de {expected_count} detectados no raw_output")
        else:
            print(f"✅ {artifacts_count} artifacts extraídos (validação por contagem não disponível)")
    
    return normalized


# === Supabase helpers ===
def get_system_message() -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    try:
        response = (
            supabase.table("system_message")
            .select("content, system_revision, ai_id, updated_at")
            .eq("is_active", True)
            .eq("agent_type", "scaffold_agent")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='scaffold_agent'")
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


def get_prd_from_message() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Busca a mensagem mais recente de prd_agent -> scaffold_agent em agent_messages
    e retorna o PRD correspondente.
    """
    try:
        response = (
            supabase.table("agent_messages")
            .select("id, prd_id, status")
            .eq("project_id", PROJECT_ID)
            .eq("from_agent", PRD_AGENT_NAME)
            .eq("to_agent", SCAFFOLD_AGENT_NAME)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar agent_messages para scaffold_agent: {exc}") from exc

    if not response.data:
        # Nenhuma mensagem pendente para este agente
        return None, None, None

    record = response.data[0]
    message_id = record.get("id")
    prd_id = record.get("prd_id")
    if not message_id or not prd_id:
        raise ValueError(
            "Mensagem encontrada em agent_messages não contém id ou prd_id válido. "
            "Verifique se o prd_agent está salvando prd_id corretamente."
        )

    # Buscar PRD correspondente em prd_documents
    try:
        prd_response = (
            supabase.table("prd_documents")
            .select("prd_id, content")
            .eq("prd_id", prd_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar PRD com prd_id={prd_id}: {exc}") from exc

    if not prd_response.data:
        raise ValueError(f"PRD com prd_id={prd_id} não encontrado em prd_documents")

    prd_record = prd_response.data[0]
    content = prd_record.get("content") or {}
    prd_payload = content.get("content") if isinstance(content, dict) else content
    raw_output = content.get("raw_output") if isinstance(content, dict) else None
    prd_string = _format_prd_payload(prd_payload, raw_output)

    return prd_string, prd_id, message_id


# === Função principal de chamada LLM ===
def call_llm(
    system_message: Optional[str],
    user_message: str,
    model: Optional[str],
    provider: Optional[str],
    system_revision: Optional[str] = None,
    max_tokens: int = 1536,
) -> dict:
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
        raise ValueError("System message não disponível para o Scaffold Agent")

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

    if not user_message:
        raise ValueError("user_message é obrigatório")

    configured_max_tokens = max_tokens
    if isinstance(CONFIG_MAX_TOKENS, int):
        configured_max_tokens = CONFIG_MAX_TOKENS
    elif configured_max_tokens is None:
        # Aumentado de 1536 para 4000 para evitar truncamento de scaffolds grandes
        configured_max_tokens = 4000

    temperature_value = CONFIG_TEMPERATURE if CONFIG_TEMPERATURE is not None else 0

    print("chamando LLM")
    raw_output: Optional[str] = None
    usage_info = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if provider == "anthropic":
        if not anthropic_client:
            raise ValueError("Anthropic não configurado. Defina ANTHROPIC_API_KEY ou ajuste o provider.")
        response = anthropic_client.messages.create(
            model=model,
            system=system_message,
            max_tokens=configured_max_tokens,
            temperature=temperature_value,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}],
                }
            ],
        )
        raw_output = "".join(part.text for part in response.content if hasattr(part, "text"))
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
    else:
        openai_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature_value,
            "max_tokens": configured_max_tokens,
        }
        if CONFIG_TOP_P is not None:
            openai_kwargs["top_p"] = CONFIG_TOP_P
        if CONFIG_FREQUENCY_PENALTY is not None:
            openai_kwargs["frequency_penalty"] = CONFIG_FREQUENCY_PENALTY
        if CONFIG_PRESENCE_PENALTY is not None:
            openai_kwargs["presence_penalty"] = CONFIG_PRESENCE_PENALTY
        if CONFIG_STOP is not None:
            openai_kwargs["stop"] = CONFIG_STOP

        response = openai_client.chat.completions.create(**openai_kwargs)
        raw_output = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)
        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    if raw_output is None:
        raise ValueError("Resposta da LLM vazia")

    # Log do tamanho da resposta para debug
    raw_output_str = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)
    print(f"📦 Raw output recebido: {len(raw_output_str)} caracteres")
    
    # Contar artifacts no raw para validação
    artifacts_in_raw = _count_artifacts_in_raw(raw_output_str)
    if artifacts_in_raw > 0:
        print(f"📊 Artifacts detectados no raw_output: {artifacts_in_raw}")
    
    # Parsear com validação rigorosa (passa raw_output_str para validação)
    normalized_content = parse_scaffold_content(raw_output, raw_output_str=raw_output_str)
    
    # Contar artifacts dos grupos (não há mais campo artifacts)
    artifacts_count = 0
    for key in ARTIFACT_KEYS:
        items = normalized_content.get(key, [])
        if isinstance(items, list):
            artifacts_count += len(items)
    print(f"✅ Artifacts normalizados: {artifacts_count} (distribuídos em grupos)")
    
    content_estimate_source = raw_output_str
    doc_tokens = len(content_estimate_source) // 4 if content_estimate_source else 0

    metadata = {
        "prompt_tokens": usage_info["prompt_tokens"],
        "completion_tokens": usage_info["completion_tokens"],
        "total_tokens": usage_info["total_tokens"],
        "agent_model": model,
        "provider": provider,
        "artifacts_count": artifacts_count,
        "scaffold_tokens": doc_tokens,
    }

    return {
        "content": normalized_content,
        "metadata": metadata,
        "raw_output": raw_output,
    }


# === Persitência ===
def save_to_scaffold_documents(result: dict, prd_id: Optional[str]) -> dict:
    if not prd_id:
        raise ValueError("prd_id inválido: é necessário para registrar scaffold_documents")

    content_jsonb = {
        "metadata": result.get("metadata"),
        "content": result.get("content"),
        "raw_output": result.get("raw_output"),
        "prd_id": prd_id,
    }

    is_using_service = supabase_service_key is not None and supabase_service_key.strip() != ""
    write_client = create_client(supabase_url, supabase_service_key) if is_using_service else supabase

    response = (
        write_client.table("scaffold_documents")
        .insert({"project_id": PROJECT_ID, "prd_id": prd_id, "content": content_jsonb})
        .execute()
    )

    if not response.data:
        raise ValueError("Nenhum registro foi inserido na tabela scaffold_documents")

    scaffold_record = response.data[0]
    scaffold_id = scaffold_record.get("scaffold_id")

    try:
        write_client.table("agent_messages")\
            .insert({
                "project_id": PROJECT_ID,
                "from_agent": SCAFFOLD_AGENT_NAME,
                "to_agent": SCHEMA_AGENT_NAME,
                "status": "pending",
                "message_content": MESSAGE_CONTENT_CREATED,
                "prd_id": prd_id,
                "scaffold_id": scaffold_id,
            })\
            .execute()
        print("log agent_messages registrado")
    except Exception as log_error:
        print(f"⚠️  Falha ao registrar mensagem em agent_messages: {log_error}")

    return scaffold_record


def build_user_message(prd_text: str, base_prompt: str) -> str:
    prompt = (base_prompt or "").strip()
    prd_section = prd_text.strip()

    sections = []
    if prompt:
        sections.append(prompt)
    sections.append("[PRD]")
    if prd_section:
        sections.append(prd_section)

    return "\n\n".join(sections)


EXPECTED_FILE_KEYS = [
    "files_root",
    "files_app",
    "files_lib",
    "files_api",
    "files_test",
]


if __name__ == "__main__":
    base_user_msg = CONFIG_USER_MESSAGE
    if not base_user_msg:
        raise ValueError(
            "Defina user_message em system/scaffold_config.json ou forneça user_message explicitamente."
        )

    message_id: Optional[str] = None

    try:
        # Recupera o PRD a partir da mensagem pendente mais recente enviada pelo prd_agent
        prd_text, prd_id, message_id = get_prd_from_message()

        if not prd_id or not prd_text or not message_id:
            print("no pending messages")
            raise SystemExit(0)

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

        user_message = build_user_message(prd_text, base_user_msg)

        system_message = CONFIG_SYSTEM_MESSAGE
        ai_model = CONFIG_AI_MODEL
        provider = CONFIG_PROVIDER
        system_revision = None

        if system_message is None or ai_model is None or provider is None:
            fetched_message, fetched_revision, fetched_model, fetched_provider = get_system_message()
            if system_message is None:
                system_message = fetched_message
            system_revision = fetched_revision
            if ai_model is None:
                ai_model = fetched_model
            if provider is None:
                provider = fetched_provider

        if system_message is None:
            raise ValueError("System message não encontrado. Configure em scaffold_config.json ou no Supabase.")

        if ai_model is None:
            ai_model = "gpt-4o"

        if provider is None:
            provider = "openai"

        print("Parâmetros efetivos da execução:")
        print(f" - model: {ai_model}")
        print(f" - provider: {provider}")
        print(f" - notes: {CONFIG_PARAMETERS.get('notes') or '<não informado>'}")

        # Aumentado max_tokens padrão para 4000 para evitar truncamento
        max_tokens_value = CONFIG_MAX_TOKENS if isinstance(CONFIG_MAX_TOKENS, int) else 4000
        
        resultado = call_llm(
            system_message=system_message,
            user_message=user_message,
            model=ai_model,
            provider=provider,
            system_revision=system_revision,
            max_tokens=max_tokens_value,
        )

        llm_meta = resultado["metadata"]
        total_tokens = llm_meta["total_tokens"]
        print(f"resposta LLM: {total_tokens} total tokens")

        saved_record = save_to_scaffold_documents(resultado, prd_id)
        scaffold_tokens = llm_meta.get("scaffold_tokens")
        print(f"scaffold salvo com sucesso: {scaffold_tokens} tokens")
        print(f"scaffold_id: {saved_record.get('scaffold_id')}")

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
        # Apenas re-levantar para encerrar sem erro adicional
        raise
    except Exception as e:
        print(f"❌ Erro na execução do scaffold_agent: {e}")
        if message_id:
            try:
                supabase.table("agent_messages")\
                    .update({
                        "status": "error",
                        "message_content": f"scaffold_error: {str(e)}",
                    })\
                    .eq("id", message_id)\
                    .eq("project_id", PROJECT_ID)\
                    .execute()
            except Exception as exc:
                print(f"⚠️  Falha ao marcar mensagem como error: {exc}")
        raise
