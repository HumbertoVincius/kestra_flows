import os
import json
import ast
import re
import time
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from llm_client import call_llm as llm_call_llm, openai_client, anthropic_client, gemini_client
from typing import Optional, Tuple, Any, Union, Iterable
from logger import init_logger, log_agent_start, log_agent_end, log_llm_call, log_llm_response, log_parse_error, log_parse_success, log_save_document, log_exception, log_info, log_error

# Carrega variáveis de ambiente do arquivo .env na raiz do projeto
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

CONFIG_PATH = Path(__file__).parent.parent / 'system' / 'prd_config.json'


def _load_prd_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open(encoding='utf-8') as config_file:
            data = json.load(config_file)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️  Falha ao ler prd_config.json: {exc}")
    return {}


def _normalize_str(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _serialize_message_payload(payload: Union[str, dict, list, None]) -> Optional[str]:
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
        if not payload:
            return None
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return "\n".join(str(item) for item in payload)

    return str(payload)


def _extract_config_params(config: dict) -> Tuple[dict, Optional[str], Optional[str], Optional[str], Optional[str]]:
    parameters = config.get("parameters") or {}
    system_payload = config.get("system_message")
    user_payload = config.get("user_message")

    ai_model = _normalize_str(parameters.get("ai_model"))
    provider = _normalize_str(parameters.get("provider"))

    normalized_system = _serialize_message_payload(system_payload)
    normalized_user = _serialize_message_payload(user_payload)

    return parameters, normalized_system, normalized_user, ai_model, provider


PRD_CONFIG = _load_prd_config()
(
    CONFIG_PARAMETERS,
    CONFIG_SYSTEM_MESSAGE,
    CONFIG_USER_MESSAGE,
    CONFIG_AI_MODEL,
    CONFIG_PROVIDER,
) = _extract_config_params(PRD_CONFIG)

CONFIG_PROVIDER = CONFIG_PROVIDER.lower() if CONFIG_PROVIDER else None
CONFIG_TEMPERATURE = CONFIG_PARAMETERS.get("temperature")
CONFIG_MAX_TOKENS = CONFIG_PARAMETERS.get("max_tokens")
CONFIG_TOP_P = CONFIG_PARAMETERS.get("top_p")
CONFIG_FREQUENCY_PENALTY = CONFIG_PARAMETERS.get("frequency_penalty")
CONFIG_PRESENCE_PENALTY = CONFIG_PARAMETERS.get("presence_penalty")
CONFIG_STOP = CONFIG_PARAMETERS.get("stop")

# === Clientes LLM ===
# Clientes importados de llm_client.py

# Inicializa o cliente Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")  # Chave anônima para leitura
supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Service role key para escrita

if not supabase_url or not supabase_key:
    # Tenta ler diretamente do arquivo como fallback
    if env_path.exists():
        content = env_path.read_text(encoding='utf-8-sig')
        for line in content.strip().split('\n'):
            if line.startswith('SUPABASE_URL=') and not supabase_url:
                supabase_url = line.split('=', 1)[1].strip()
            elif line.startswith('SUPABASE_KEY=') and not supabase_key:
                supabase_key = line.split('=', 1)[1].strip()
            elif line.startswith('SUPABASE_SERVICE_ROLE_KEY=') and not supabase_service_key:
                supabase_service_key = line.split('=', 1)[1].strip()

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e/ou SUPABASE_KEY não encontradas no arquivo .env")

# Cliente com chave anônima para leitura
supabase: Client = create_client(supabase_url, supabase_key)

# Cliente com service role key para escrita (bypassa RLS)
# Se não houver service role key, tenta usar anon key para escrita
# Nota: supabase_write será recriado na função save_to_prd_documents para garantir uso correto
supabase_write: Client = None

PROJECT_ID = "639e810b-9d8c-4f31-9569-ecf61fb43888"
PRD_AGENT_NAME = "prd_agent"
SCAFFOLD_AGENT_NAME = "scaffold_agent"
ANALYZER_AGENT_NAME = "analyzer_agent"
MESSAGE_CONTENT_CREATED = "prd_created"


def get_system_message() -> Tuple[str, str, Optional[str], Optional[str]]:
    
    try:
        response = supabase.table("system_message")\
            .select("content, system_revision, ai_id, updated_at")\
            .eq("is_active", True)\
            .eq("agent_type", "prd_agent")\
            .order("updated_at", desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data or len(response.data) == 0:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='prd_agent'")
        
        record = response.data[0]
        content = record.get("content")
        revision = record.get("system_revision", "")
        ai_id = record.get("ai_id")
        ai_model = None
        ai_provider = None
        
        if not content:
            raise ValueError("Campo 'content' está vazio no registro encontrado")
        
        if ai_id:
            ai_response = supabase.table("ai_models")\
                .select("ai_model, provider")\
                .eq("ai_id", ai_id)\
                .limit(1)\
                .execute()
            if ai_response.data:
                ai_model = ai_response.data[0].get("ai_model")
                ai_provider = ai_response.data[0].get("provider")
        
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, dict):
            content_str = str(content) if content else ""
        else:
            content_str = str(content)
        
        return content_str, revision, ai_model, ai_provider
        
    except Exception as e:
        print(f"Erro ao buscar system message do Supabase: {e}")
        raise


def get_latest_codegen_id() -> Optional[str]:
    """
    Busca o ID do último codegen gerado para o projeto.
    """
    try:
        response = (
            supabase.table("codegen_documents")
            .select("codegen_id")
            .eq("project_id", PROJECT_ID)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if response.data:
            return response.data[0].get("codegen_id")
    except Exception as exc:
        print(f"⚠️  Erro ao buscar último codegen: {exc}")
    
    return None


def detect_input_type(user_input: str) -> str:
    """
    Detecta se o input do usuário é um PRD (modificação incremental) ou um erro/bug reportado.
    Usa heurísticas simples primeiro, depois LLM leve se necessário.
    
    Returns:
        "prd" ou "error"
    """
    if not user_input or not user_input.strip():
        return "prd"  # Default para PRD se vazio
    
    input_lower = user_input.lower()
    
    # Heurísticas simples para detectar erros
    error_patterns = [
        r'\berror\b',
        r'\bexception\b',
        r'\bthrow\b',
        r'\bfailed\b',
        r'\bfailure\b',
        r'\bbug\b',
        r'\berro\b',
        r'\bfalha\b',
        r'at\s+\w+\.\w+',  # Stack trace pattern: "at Class.method"
        r'\.tsx?:\d+:\d+',  # File:line:column pattern
        r'\.jsx?:\d+:\d+',
        r'TypeError',
        r'ReferenceError',
        r'SyntaxError',
        r'Cannot find',
        r'is not defined',
        r'is not exported',
        r'Module not found',
        r'stack trace',
        r'traceback',
    ]
    
    # Verificar padrões de erro
    for pattern in error_patterns:
        if re.search(pattern, input_lower, re.IGNORECASE):
            print(f"🔍 Detectado padrão de erro: {pattern}")
            return "error"
    
    # Se não encontrou padrões claros, usar LLM leve para classificar
    print("🔍 Padrões não conclusivos, usando LLM para classificar...")
    
    classification_prompt = f"""Analise o seguinte input do usuário e classifique como:
- "prd" se for uma solicitação de modificação incremental, novo requisito ou feature
- "error" se for um relatório de bug, erro de runtime, stack trace ou problema técnico

Input:
{user_input[:1000]}

Responda APENAS com "prd" ou "error", sem explicações."""

    try:
        # Usar LLM rápido e barato para classificação
        result = llm_call_llm(
            system_message="Você é um classificador de inputs. Analise e responda apenas com 'prd' ou 'error'.",
            user_message=classification_prompt,
            model="gpt-4o-mini",  # Modelo mais barato para classificação
            provider="openai",
            max_tokens=10,
            default_max_tokens=10,
            default_temperature=0,
            agent_name="PRD Classifier",
        )
        
        classification = result["raw_output"].strip().lower()
        if "error" in classification:
            return "error"
        else:
            return "prd"
    except Exception as exc:
        print(f"⚠️  Erro ao classificar com LLM: {exc}. Assumindo PRD por padrão.")
        return "prd"


def create_error_message_for_analyzer(error_text: str, codegen_id: Optional[str] = None) -> bool:
    """
    Cria uma mensagem em agent_messages para o Analyzer Agent processar um erro reportado.
    
    Args:
        error_text: Texto do erro reportado pelo usuário
        codegen_id: ID do codegen mais recente (se None, busca automaticamente)
    
    Returns:
        True se sucesso, False caso contrário
    """
    print("🔧 Criando mensagem de erro para Analyzer Agent...")
    
    # Buscar codegen_id se não fornecido
    if not codegen_id:
        codegen_id = get_latest_codegen_id()
        if not codegen_id:
            print("❌ ERRO: Nenhum codegen encontrado. Execute o codegen primeiro.")
            return False
        print(f"📦 Usando último codegen: {codegen_id}")
    
    # Criar message_content estruturado
    message_content = json.dumps({
        "type": "error_report",
        "error_text": error_text[:5000],  # Limitar tamanho
        "codegen_id": codegen_id,
        "source": "prd_agent"
    }, ensure_ascii=False)
    
    # Verificar se há service key para escrita
    is_using_service_key = supabase_service_key is not None and supabase_service_key.strip() != ""
    write_client = create_client(supabase_url, supabase_service_key) if is_using_service_key else supabase
    
    try:
        msg_doc = {
            "project_id": PROJECT_ID,
            "from_agent": PRD_AGENT_NAME,
            "to_agent": ANALYZER_AGENT_NAME,
            "status": "pending",
            "message_content": message_content,
            "codegen_id": codegen_id,
        }
        
        result = write_client.table("agent_messages").insert(msg_doc).execute()
        
        if result.data:
            print(f"✅ Mensagem de erro criada com sucesso!")
            print(f"   - Message ID: {result.data[0].get('id')}")
            print(f"   - Codegen ID: {codegen_id}")
            print(f"\n📝 Próximo passo: Execute o analyzer para processar o erro.")
            return True
        else:
            print("❌ Erro: Mensagem não foi criada")
            return False
            
    except Exception as exc:
        print(f"❌ Erro ao criar mensagem de erro: {exc}")
        return False


def _extract_code_fence(raw_str: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_str, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw_str


def _extract_first_json_object(raw_str: str) -> Optional[str]:
    start = raw_str.find('{')
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
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return raw_str[start:idx + 1]

    return None


def _parse_jsonish(raw_str: str) -> Any:
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

    first_object = _extract_first_json_object(raw_str)
    if first_object:
        try:
            return json.loads(first_object)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(first_object)
            except (ValueError, SyntaxError):
                return None

    return None


def _normalize_artifact_entry(entry: Any) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(f"Artifact inválido: {entry}")

    path = entry.get("path")
    content = entry.get("content")

    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"Artifact sem path válido: {entry}")

    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    elif content is None:
        content = ""
    else:
        content = str(content)

    return {"path": path.strip(), "content": content}


def _normalize_artifact_lists(payload: dict) -> dict:
    artifact_keys = [
        "artifacts",
        "files",
        "file_entries",
        "fileEntries",
        "files_root",
        "files_app",
        "files_lib",
        "files_api",
        "files_test",
    ]

    normalized_payload = deepcopy(payload)
    aggregated = []

    for key in artifact_keys:
        if key not in normalized_payload:
            continue
        value = normalized_payload[key]
        if isinstance(value, list):
            normalized_list = [_normalize_artifact_entry(item) for item in value]
            normalized_payload[key] = normalized_list
            aggregated.extend(normalized_list)

    if aggregated:
        normalized_payload["artifacts"] = aggregated

    return normalized_payload


def parse_prd_content(raw: Any) -> dict:
    """Normaliza a resposta da LLM para um dicionário JSON serializável."""
    if isinstance(raw, dict):
        parsed = raw
    else:
        if raw is None:
            raise ValueError("Resposta da LLM vazia para o PRD")
        raw_str = str(raw).strip()
        if not raw_str:
            raise ValueError("Resposta da LLM vazia para o PRD")

        raw_str = _extract_code_fence(raw_str)
        parsed = _parse_jsonish(raw_str)
        if parsed is None:
            raise ValueError(
                "Não foi possível converter a resposta da LLM em JSON válido. Trecho inicial: "
                + raw_str[:200]
            )

    if isinstance(parsed, list):
        normalized = {"artifacts": [_normalize_artifact_entry(item) for item in parsed]}
    elif isinstance(parsed, dict):
        data = deepcopy(parsed)

        if "prd" in data:
            prd_value = data["prd"]
            if isinstance(prd_value, str):
                inner = _parse_jsonish(_extract_code_fence(prd_value.strip()))
                if isinstance(inner, dict):
                    prd_value = inner
                else:
                    raise ValueError("Não foi possível normalizar a chave 'prd' da resposta da LLM")
            elif not isinstance(prd_value, dict):
                raise ValueError("A chave 'prd' deve ser um objeto JSON")

            content_dict = deepcopy(prd_value)
            for key, value in data.items():
                if key != "prd" and key not in content_dict:
                    content_dict[key] = value
            normalized = content_dict
        else:
            normalized = data

        normalized = _normalize_artifact_lists(normalized)
    else:
        raise ValueError("Resposta da LLM não contém um objeto JSON válido")

    try:
        sanitized = json.loads(json.dumps(normalized))
    except (TypeError, ValueError) as exc:
        raise ValueError("Conteúdo do PRD contém tipos não serializáveis") from exc

    return sanitized


def call_llm(
    system_message: str = None,
    system_revision: str = None,
    user_message: str = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_tokens: int = 1536
):
    start_time = time.time()
    try:
        # Overrides vindos do arquivo de configuração
        if system_message is None:
            system_message = CONFIG_SYSTEM_MESSAGE
        if user_message is None:
            user_message = CONFIG_USER_MESSAGE
        
        model_used = model or CONFIG_AI_MODEL or "unknown"
        provider_used = provider or CONFIG_PROVIDER or "unknown"
        
        # Logar chamada LLM
        log_llm_call(
            "Chamando LLM para gerar PRD",
            model_used,
            provider_used
        )
        
        result = llm_call_llm(
            system_message=system_message,
            user_message=user_message,
            model=model or CONFIG_AI_MODEL,
            provider=provider or CONFIG_PROVIDER,
            system_revision=system_revision,
            max_tokens=max_tokens,
            default_max_tokens=1536,
            default_temperature=0,
            agent_name="PRD Agent",
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
        
        prd_content_raw = result["raw_output"]
        usage_info = result["metadata"]
        
        if prd_content_raw is None:
            raise ValueError("Resposta da LLM vazia")
        
        # Logar resposta LLM
        model_used = model or CONFIG_AI_MODEL or "unknown"
        provider_used = provider or CONFIG_PROVIDER or "unknown"
        log_llm_response(
            "Resposta LLM recebida para PRD",
            model_used,
            provider_used,
            tokens_used=usage_info.get("total_tokens"),
            raw_response=str(prd_content_raw)[:5000] if isinstance(prd_content_raw, str) else str(prd_content_raw)[:5000]
        )
        
        # Normaliza o conteúdo antes de prosseguir
        try:
            normalized_content = parse_prd_content(prd_content_raw)
            log_parse_success("PRD parseado com sucesso", parsed_size=len(str(normalized_content)))
        except ValueError as parse_err:
            log_parse_error(
                "Erro ao parsear resposta PRD",
                str(parse_err),
                raw_data=str(prd_content_raw)[:10000] if isinstance(prd_content_raw, str) else str(prd_content_raw)[:10000]
            )
            raise
        
        # Estima o tamanho do PRD em tokens (aproximação: 1 token ≈ 4 caracteres)
        content_estimate_source = prd_content_raw if isinstance(prd_content_raw, str) else json.dumps(prd_content_raw)
        prd_tokens = len(content_estimate_source) // 4 if content_estimate_source else 0
        
        artifact_count = 0
        artifact_characters = 0
        if isinstance(normalized_content, dict):
            artifacts = normalized_content.get("artifacts")
            if isinstance(artifacts, list):
                artifact_count = len(artifacts)
                for artifact in artifacts:
                    body = artifact.get("content")
                    if isinstance(body, str):
                        artifact_characters += len(body)

        # Monta o JSON de metadados
        metadata = {
            "prompt_tokens": usage_info.get("prompt_tokens", 0),
            "completion_tokens": usage_info.get("completion_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0),
            "agent_model": usage_info.get("agent_model", model),
            "provider": usage_info.get("provider", provider),
            "prd_tokens": prd_tokens,
            "artifact_count": artifact_count,
            "artifact_characters": artifact_characters,
        }
        
        # Calcular tempo de execução
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # Logar resposta com tempo
        log_llm_response(
            "PRD gerado com sucesso",
            model_used,
            provider_used,
            tokens_used=usage_info.get("total_tokens"),
            execution_time_ms=execution_time_ms
        )
        
        # Retorna a resposta e os metadados
        return {
            "content": normalized_content,
            "metadata": metadata,
            "raw_output": prd_content_raw
        }
    
    except Exception as e:
        log_exception("error", "Erro ao chamar LLM para PRD", e)
        print(f"Erro ao chamar a LLM: {e}")
        raise


def save_to_prd_documents(result: dict):
    try:
        # Monta o JSONB com content normalizado e metadata
        content_jsonb = {
            "metadata": result.get("metadata"),
            "content": result.get("content"),
            "raw_output": result.get("raw_output"),
        }
        
        # Logar início do salvamento
        log_info("save_document", "Iniciando salvamento de PRD")

        # Verifica qual cliente está sendo usado
        is_using_service_key = supabase_service_key is not None and supabase_service_key.strip() != ""

        # Garante que estamos usando o cliente correto
        # Recria o cliente para garantir que está usando a chave correta
        if is_using_service_key:
            write_client = create_client(supabase_url, supabase_service_key)
        else:
            write_client = supabase

        # Insere na tabela prd_documents
        response = write_client.table("prd_documents")\
            .insert({
                "project_id": PROJECT_ID,
                "content": content_jsonb
            })\
            .execute()

        if not response.data or len(response.data) == 0:
            raise ValueError("Nenhum registro foi inserido na tabela prd_documents")

        prd_record = response.data[0]
        prd_id = prd_record.get("prd_id")
        
        # Logar salvamento bem-sucedido
        log_save_document(
            "PRD salvo com sucesso",
            "prd",
            document_id=prd_id,
            prd_id=prd_id
        )

        try:
            write_client.table("agent_messages")\
                .insert({
                    "project_id": PROJECT_ID,
                    "from_agent": PRD_AGENT_NAME,
                    "to_agent": ANALYZER_AGENT_NAME,
                    "status": "pending",
                    "message_content": MESSAGE_CONTENT_CREATED,
                    "prd_id": prd_id,
                })\
                .execute()
            print("log agent_messages registrado")
            log_info("message_sent", f"Mensagem enviada para {ANALYZER_AGENT_NAME}", prd_id=prd_id)
        except Exception as log_error:
            print(f"⚠️  Falha ao registrar mensagem em agent_messages: {log_error}")
            log_exception("error", "Erro ao registrar mensagem em agent_messages", log_error, prd_id=prd_id)

        return prd_record

    except Exception as e:
        log_exception("error", "Erro ao salvar PRD no Supabase", e)
        error_msg = str(e)
        print(f"\n❌ Erro ao salvar no Supabase:")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensagem: {error_msg}")

        # Tenta extrair informações detalhadas do erro
        if hasattr(e, 'args') and len(e.args) > 0:
            error_dict = e.args[0] if isinstance(e.args[0], dict) else {}
            if isinstance(error_dict, dict):
                print(f"   Código: {error_dict.get('code', 'N/A')}")
                print(f"   Detalhes: {error_dict.get('details', 'N/A')}")
                print(f"   Hint: {error_dict.get('hint', 'N/A')}")

        # Verificações adicionais
        print(f"\n🔍 Diagnóstico:")
        has_service_key = supabase_service_key is not None and supabase_service_key.strip() != ""
        print(f"   - Service role key configurada: {'✅ Sim' if has_service_key else '❌ Não'}")
        if has_service_key:
            print(f"   - Service role key (primeiros 30 chars): {supabase_service_key[:30]}...")
            print(f"   - Verifique se a service role key está correta no .env")
            print(f"   - Verifique se não há constraints ou triggers bloqueando")
            print(f"   - Service role key DEVE bypassar RLS - se ainda falhar, há outro problema")
        else:
            print(f"   ✅ SOLUÇÃO: Configure SUPABASE_SERVICE_ROLE_KEY no arquivo .env")
            print(f"      (Encontre em: Supabase Dashboard → Settings → API → Service Role Key)")
            print(f"      (Service role key bypassa RLS completamente)")
            print(f"      (Adicione no .env: SUPABASE_SERVICE_ROLE_KEY=sua_chave_aqui)")

        raise


if __name__ == "__main__":
    # Inicializar logger
    init_logger(PRD_AGENT_NAME, CONFIG_PARAMETERS)
    agent_start_time = time.time()
    log_agent_start("Iniciando execução do PRD Agent")
    
    # Carrega mensagens e parâmetros a partir do arquivo de configuração
    user_msg = CONFIG_USER_MESSAGE
    system_msg = CONFIG_SYSTEM_MESSAGE
    system_model = CONFIG_AI_MODEL
    system_provider = CONFIG_PROVIDER
    system_rev: Optional[str] = None

    if not user_msg:
        raise ValueError(
            "Defina user_message em system/prd_config.json ou forneça user_message explicitamente."
        )

    # Detectar tipo de input (PRD ou erro)
    print("🔍 Analisando tipo de input do usuário...")
    input_type = detect_input_type(user_msg)
    
    if input_type == "error":
        print("🐛 Input detectado como ERRO/BUG - roteando para Analyzer Agent")
        log_info("message_sent", "Roteando erro para Analyzer Agent")
        success = create_error_message_for_analyzer(user_msg)
        if success:
            print("✅ Erro roteado com sucesso para Analyzer Agent")
            print("   Execute o analyzer para processar o erro e gerar relatório de impacto.")
            log_info("message_sent", "Erro roteado com sucesso para Analyzer Agent")
        else:
            print("❌ Falha ao rotear erro para Analyzer Agent")
            log_error("error", "Falha ao rotear erro para Analyzer Agent")
        # Encerrar sem processar como PRD
        execution_time_ms = int((time.time() - agent_start_time) * 1000)
        log_agent_end("Execução do PRD Agent concluída (modo erro)", execution_time_ms=execution_time_ms)
        raise SystemExit(0)
    
    print("📋 Input detectado como PRD - processando normalmente")
    log_info("info", "Input detectado como PRD - processando normalmente")
    
    # Fallback para Supabase quando informações essenciais não estão no arquivo local
    if system_msg is None or system_model is None or system_provider is None:
        fetched_msg, fetched_rev, fetched_model, fetched_provider = get_system_message()
        if system_msg is None:
            system_msg = fetched_msg
        system_rev = fetched_rev
        if system_model is None:
            system_model = fetched_model
        if system_provider is None:
            system_provider = fetched_provider

    if system_msg is None:
        raise ValueError("System message não encontrado. Configure em prd_config.json ou no Supabase.")

    if system_model is None:
        system_model = "gpt-4o"

    if system_provider is None:
        system_provider = "openai"

    print("Parâmetros efetivos da execução:")
    print(f" - model: {system_model}")
    print(f" - provider: {system_provider}")
    print(f" - parameters.notes: {CONFIG_PARAMETERS.get('notes') or '<não informado>'}")

    # Chama LLM
    resultado = call_llm(
        system_message=system_msg,
        system_revision=system_rev,
        user_message=user_msg,
        model=system_model,
        provider=system_provider
    )
    llm_meta = resultado['metadata']
    total_tokens = llm_meta['total_tokens']
    print(f"resposta LLM: {total_tokens} total tokens")

    # Salva no banco
    saved_record = save_to_prd_documents(resultado)
    prd_tokens = llm_meta['prd_tokens']
    print(f"prd salvo com sucesso: {prd_tokens} tokens")
    
    # Logar fim da execução
    execution_time_ms = int((time.time() - agent_start_time) * 1000)
    log_agent_end("Execução do PRD Agent concluída com sucesso", execution_time_ms=execution_time_ms, prd_id=saved_record.get("prd_id"))

