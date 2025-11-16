import os
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client


# === Caminhos e configuração ===
ENV_PATH = Path(__file__).parent.parent / ".env"
CONFIG_PATH = Path(__file__).parent.parent / "system" / "schema_config.json"

load_dotenv(dotenv_path=ENV_PATH)


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️  Falha ao ler schema_config.json: {exc}")
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


SCHEMA_CONFIG = _load_config()
(
    CONFIG_PARAMETERS,
    CONFIG_SYSTEM_MESSAGE,
    CONFIG_USER_MESSAGE,
    CONFIG_AI_MODEL,
    CONFIG_PROVIDER,
) = _extract_config_values(SCHEMA_CONFIG)

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
SCHEMA_AGENT_NAME = "schema_agent"
SCAFFOLD_AGENT_NAME = "scaffold_agent"
CODEGEN_AGENT_NAME = "codegen_agent"
MESSAGE_CONTENT_CREATED = "schema_created"


# === Supabase helpers ===
def get_system_message() -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    try:
        response = (
            supabase.table("system_message")
            .select("content, system_revision, ai_id, updated_at")
            .eq("is_active", True)
            .eq("agent_type", "schema_agent")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='schema_agent'")
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


def _format_prd_text(content: Any) -> str:
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            return str(content)
    if isinstance(content, list):
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            return "\n".join(str(x) for x in content)
    if isinstance(content, str):
        return content
    return str(content)


def _format_scaffold_text(content: Any) -> str:
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            return str(content)
    if isinstance(content, list):
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            return "\n".join(str(x) for x in content)
    if isinstance(content, str):
        return content
    return str(content)


def get_prd_and_scaffold_from_message() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Busca a mensagem pendente mais recente de scaffold_agent -> schema_agent
    e carrega PRD e Scaffold correspondentes.
    """
    try:
        response = (
            supabase.table("agent_messages")
            .select("id, prd_id, scaffold_id, status")
            .eq("project_id", PROJECT_ID)
            .eq("from_agent", SCAFFOLD_AGENT_NAME)
            .eq("to_agent", SCHEMA_AGENT_NAME)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar agent_messages para schema_agent: {exc}") from exc

    if not response.data:
        # Nenhuma mensagem pendente
        return None, None, None, None, None

    msg = response.data[0]
    message_id = msg.get("id")
    prd_id = msg.get("prd_id")
    scaffold_id = msg.get("scaffold_id")

    if not message_id or not prd_id or not scaffold_id:
        raise ValueError(
            "Mensagem encontrada em agent_messages não contém id, prd_id ou scaffold_id válidos. "
            "Verifique se o scaffold_agent está salvando esses campos corretamente."
        )

    # Buscar PRD
    try:
        prd_resp = (
            supabase.table("prd_documents")
            .select("content")
            .eq("prd_id", prd_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar PRD com prd_id={prd_id}: {exc}") from exc

    if not prd_resp.data:
        raise ValueError(f"PRD com prd_id={prd_id} não encontrado em prd_documents")

    prd_content_raw = prd_resp.data[0].get("content") or {}
    prd_payload = prd_content_raw.get("content") if isinstance(prd_content_raw, dict) else prd_content_raw
    prd_text = _format_prd_text(prd_payload)

    # Buscar Scaffold
    try:
        scaffold_resp = (
            supabase.table("scaffold_documents")
            .select("content")
            .eq("scaffold_id", scaffold_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar Scaffold com scaffold_id={scaffold_id}: {exc}") from exc

    if not scaffold_resp.data:
        raise ValueError(f"Scaffold com scaffold_id={scaffold_id} não encontrado em scaffold_documents")

    scaffold_content_raw = scaffold_resp.data[0].get("content") or {}
    scaffold_payload = scaffold_content_raw.get("content") if isinstance(scaffold_content_raw, dict) else scaffold_content_raw
    scaffold_text = _format_scaffold_text(scaffold_payload)

    return prd_text, scaffold_text, prd_id, scaffold_id, message_id


# === LLM ===
def build_user_message(prd_text: str, scaffold_text: str, base_prompt: str) -> str:
    sections = []

    if base_prompt:
        sections.append(base_prompt.strip())

    if prd_text:
        sections.append("[PRD]")
        sections.append(prd_text.strip())

    if scaffold_text:
        sections.append("[SCAFFOLD]")
        sections.append(scaffold_text.strip())

    return "\n\n".join(sections)


def call_llm(
    system_message: Optional[str],
    user_message: str,
    model: Optional[str],
    provider: Optional[str],
    system_revision: Optional[str] = None,
    max_tokens: int = 4000,
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
        raise ValueError("System message não disponível para o Schema Agent")

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
        configured_max_tokens = 4000

    temperature_value = CONFIG_TEMPERATURE if CONFIG_TEMPERATURE is not None else 0.1

    print("chamando LLM (schema_agent)")
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
        raise ValueError("Resposta da LLM vazia para o Schema Agent")

    raw_str = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)
    print(f"📦 Raw output (schema_agent): {len(raw_str)} caracteres")

    parsed = _parse_schema_content(raw_str)

    metadata = {
        "prompt_tokens": usage_info["prompt_tokens"],
        "completion_tokens": usage_info["completion_tokens"],
        "total_tokens": usage_info["total_tokens"],
        "agent_model": model,
        "provider": provider,
        "system_revision": system_revision or "",
    }

    return {
        "content": parsed,
        "metadata": metadata,
        "raw_output": raw_output,
    }


def _parse_schema_content(raw_str: str) -> dict:
    try:
        parsed = json.loads(raw_str)
    except json.JSONDecodeError:
        raise ValueError(
            "Não foi possível converter a resposta da LLM em JSON válido para o schema. "
            f"Trecho inicial: {raw_str[:400]}"
        )

    if not isinstance(parsed, dict):
        raise ValueError("Resposta do schema_agent não é um objeto JSON.")

    migration_sql = parsed.get("migration_sql") or ""
    schema_summary = parsed.get("schema_summary") or {}

    if not isinstance(migration_sql, str):
        raise ValueError("Campo migration_sql deve ser string.")
    if not isinstance(schema_summary, dict):
        raise ValueError("Campo schema_summary deve ser objeto.")

    return {
        "migration_sql": migration_sql,
        "schema_summary": schema_summary,
    }


def save_to_schema_documents(result: dict, prd_id: Optional[str], scaffold_id: Optional[str]) -> dict:
    if not prd_id or not scaffold_id:
        raise ValueError("prd_id e scaffold_id são obrigatórios para registrar schema_documents")

    content_jsonb = {
        "metadata": result.get("metadata"),
        "migration_sql": result.get("content", {}).get("migration_sql", ""),
        "schema_summary": result.get("content", {}).get("schema_summary", {}),
        "raw_output": result.get("raw_output"),
    }

    is_using_service = supabase_service_key is not None and supabase_service_key.strip() != ""
    write_client = create_client(supabase_url, supabase_service_key) if is_using_service else supabase

    insert_payload = {
        "project_id": PROJECT_ID,
        "prd_id": prd_id,
        "scaffold_id": scaffold_id,
        "content": content_jsonb,
    }

    response = write_client.table("schema_documents").insert(insert_payload).execute()
    if not response.data:
        raise ValueError("Nenhum registro foi inserido na tabela schema_documents")

    schema_record = response.data[0]
    schema_id = schema_record.get("schema_id")

    try:
        write_client.table("agent_messages")\
            .insert({
                "project_id": PROJECT_ID,
                "from_agent": SCHEMA_AGENT_NAME,
                "to_agent": CODEGEN_AGENT_NAME,
                "status": "pending",
                "message_content": MESSAGE_CONTENT_CREATED,
                "prd_id": prd_id,
                "scaffold_id": scaffold_id,
                "schema_id": schema_id,
            })\
            .execute()
        print("log agent_messages (schema_agent → codegen_agent) registrado")
    except Exception as log_error:
        print(f"⚠️  Falha ao registrar mensagem em agent_messages: {log_error}")

    return schema_record


if __name__ == "__main__":
    base_user_msg = CONFIG_USER_MESSAGE
    if not base_user_msg:
        raise ValueError(
            "Defina user_message em system/schema_config.json ou forneça user_message explicitamente."
        )

    system_msg = CONFIG_SYSTEM_MESSAGE
    ai_model = CONFIG_AI_MODEL
    provider = CONFIG_PROVIDER
    system_rev: Optional[str] = None
    message_id: Optional[str] = None

    try:
        # Carregar PRD + Scaffold a partir de mensagem pendente
        prd_text, scaffold_text, prd_id, scaffold_id, message_id = get_prd_and_scaffold_from_message()

        if not prd_text or not scaffold_text or not prd_id or not scaffold_id or not message_id:
            print("no pending messages")
            raise SystemExit(0)

        # Marcar mensagem como processing
        try:
            supabase.table("agent_messages")\
                .update({"status": "processing"})\
                .eq("id", message_id)\
                .execute()
        except Exception as exc:
            print(f"⚠️  Falha ao marcar mensagem como processing: {exc}")

        # Completar config a partir do Supabase, se necessário
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
            raise ValueError("System message não encontrado. Configure em schema_config.json ou no Supabase.")

        if ai_model is None:
            ai_model = "gpt-4o"

        if provider is None:
            provider = "openai"

        print("Parâmetros efetivos da execução (schema_agent):")
        print(f" - model: {ai_model}")
        print(f" - provider: {provider}")
        print(f" - notes: {CONFIG_PARAMETERS.get('notes') or '<não informado>'}")

        max_tokens_value = CONFIG_MAX_TOKENS if isinstance(CONFIG_MAX_TOKENS, int) else 4000

        user_message = build_user_message(prd_text, scaffold_text, base_user_msg)

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
        print(f"resposta LLM (schema_agent): {total_tokens} total tokens")

        saved_record = save_to_schema_documents(resultado, prd_id, scaffold_id)
        schema_id = saved_record.get("schema_id")
        print(f"schema salvo com sucesso: schema_id={schema_id}")

        # Marcar mensagem original como ok
        try:
            supabase.table("agent_messages")\
                .update({"status": "ok"})\
                .eq("id", message_id)\
                .execute()
        except Exception as exc:
            print(f"⚠️  Falha ao marcar mensagem como ok: {exc}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ Erro na execução do schema_agent: {e}")
        if message_id:
            try:
                supabase.table("agent_messages")\
                    .update({
                        "status": "error",
                        "message_content": f"schema_error: {str(e)}",
                    })\
                    .eq("id", message_id)\
                    .execute()
            except Exception as exc:
                print(f"⚠️  Falha ao marcar mensagem como error: {exc}")
        raise


