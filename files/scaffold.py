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


# === Utilitários de parsing ===
def _extract_code_fence(raw_str: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_str, re.IGNORECASE)
    return match.group(1).strip() if match else raw_str


def _extract_first_json_object(raw_str: str) -> Optional[str]:
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


ARTIFACT_KEYS = [
    "files_root",
    "files_app",
    "files_lib",
    "files_api",
    "files_test",
]


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
    result: dict[str, Any] = {}
    artifact_groups = {key: [] for key in ARTIFACT_KEYS}

    if isinstance(data, list):
        artifact_groups["files_root"] = [_normalize_artifact_entry(item) for item in data]
        return {key: value for key, value in artifact_groups.items() if value}

    if not isinstance(data, dict):
        raise ValueError("Estrutura retornada pela LLM não é lista nem objeto")

    for key, value in data.items():
        if key == "artifacts" and isinstance(value, list):
            artifact_groups["files_root"].extend(_normalize_artifact_entry(item) for item in value)
        elif key in ARTIFACT_KEYS and isinstance(value, list):
            artifact_groups[key] = [_normalize_artifact_entry(item) for item in value]
        else:
            result[key] = value

    for key, items in artifact_groups.items():
        if items:
            result[key] = items

    return result


# === LLM Parsing ===
def parse_scaffold_content(raw: Any) -> dict:
    if isinstance(raw, dict):
        parsed = raw
    else:
        if raw is None:
            raise ValueError("Resposta da LLM vazia para o Scaffold")
        raw_str = str(raw).strip()
        if not raw_str:
            raise ValueError("Resposta da LLM vazia para o Scaffold")
        raw_str = _extract_code_fence(raw_str)
        parsed = _parse_jsonish(raw_str)
        if parsed is None:
            raise ValueError(
                "Não foi possível converter a resposta da LLM em JSON válido. Trecho inicial: "
                + raw_str[:200]
            )
    if isinstance(parsed, list):
        return _normalize_scaffold_structure(parsed)
    if isinstance(parsed, dict):
        return _normalize_scaffold_structure(parsed)
    raise ValueError("Resposta da LLM não contém um objeto JSON válido")


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


def get_latest_prd() -> Tuple[str, Optional[str]]:
    order_columns = ["inserted_at", "created_at", "updated_at", None]
    errors = []
    for column in order_columns:
        try:
            query = supabase.table("prd_documents").select("prd_id, content")
            if column:
                query = query.select(f"prd_id, content, {column}")
                query = query.order(column, desc=True)
            response = query.limit(1).execute()
            if response.data:
                record = response.data[0]
                content = record.get("content") or {}
                prd_id = record.get("prd_id")
                prd_payload = content.get("content") if isinstance(content, dict) else content
                prd_string = json.dumps(prd_payload, ensure_ascii=False, indent=2)
                return prd_string, prd_id
        except Exception as exc:
            errors.append((column or "<sem ordenação>", str(exc)))
            continue
    for column, message in errors:
        print(f"⚠️  Falha ao ordenar por '{column}': {message}")
    raise ValueError("Não foi possível recuperar o PRD mais recente de prd_documents")


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
        configured_max_tokens = 1536

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

    normalized_content = parse_scaffold_content(raw_output)
    content_estimate_source = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
    doc_tokens = len(content_estimate_source) // 4 if content_estimate_source else 0

    metadata = {
        "prompt_tokens": usage_info["prompt_tokens"],
        "completion_tokens": usage_info["completion_tokens"],
        "total_tokens": usage_info["total_tokens"],
        "agent_model": model,
        "provider": provider,
        "system_revision": system_revision or "",
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
                "to_agent": CODEGEN_AGENT_NAME,
                "status": MESSAGE_CONTENT_CREATED,
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
    prompt = base_prompt or ""
    sections = [prompt.strip(), "\n\n[PRD]", prd_text.strip()]
    return "\n".join(part for part in sections if part)


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

    prd_text, prd_id = get_latest_prd()
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

    resultado = call_llm(
        system_message=system_message,
        user_message=user_message,
        model=ai_model,
        provider=provider,
        system_revision=system_revision,
        max_tokens=CONFIG_MAX_TOKENS or 1536,
    )

    llm_meta = resultado["metadata"]
    total_tokens = llm_meta["total_tokens"]
    print(f"resposta LLM: {total_tokens} total tokens")

    saved_record = save_to_scaffold_documents(resultado, prd_id)
    scaffold_tokens = llm_meta.get("scaffold_tokens")
    print(f"scaffold salvo com sucesso: {scaffold_tokens} tokens")
    print(f"scaffold_id: {saved_record.get('scaffold_id')}")
