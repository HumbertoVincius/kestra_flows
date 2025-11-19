import os
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union, List, Dict

from dotenv import load_dotenv
from supabase import Client, create_client
from llm_client import call_llm as llm_call_llm, openai_client, anthropic_client, gemini_client

# === Caminhos e configuração ===
ENV_PATH = Path(__file__).parent.parent / ".env"
CONFIG_PATH = Path(__file__).parent.parent / "system" / "analyzer_config.json"

load_dotenv(dotenv_path=ENV_PATH)


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️  Falha ao ler analyzer_config.json: {exc}")
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


ANALYZER_CONFIG = _load_config()
(
    CONFIG_PARAMETERS,
    CONFIG_SYSTEM_MESSAGE,
    CONFIG_USER_MESSAGE,
    CONFIG_AI_MODEL,
    CONFIG_PROVIDER,
) = _extract_config_values(ANALYZER_CONFIG)

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
ANALYZER_AGENT_NAME = "analyzer_agent"
PRD_AGENT_NAME = "prd_agent"
SCAFFOLD_AGENT_NAME = "scaffold_agent"
MESSAGE_CONTENT_CREATED = "impact_report_created"


# === Supabase helpers ===
def get_system_message() -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    try:
        response = (
            supabase.table("system_message")
            .select("content, system_revision, ai_id, updated_at")
            .eq("is_active", True)
            .eq("agent_type", "analyzer_agent")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='analyzer_agent'")
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


def get_new_prd_from_message() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Busca a mensagem mais recente de prd_agent -> analyzer_agent em agent_messages
    e retorna o PRD novo correspondente.
    """
    try:
        response = (
            supabase.table("agent_messages")
            .select("id, prd_id, status")
            .eq("project_id", PROJECT_ID)
            .eq("from_agent", PRD_AGENT_NAME)
            .eq("to_agent", ANALYZER_AGENT_NAME)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar agent_messages para analyzer_agent: {exc}") from exc

    if not response.data:
        return None, None, None

    record = response.data[0]
    message_id = record.get("id")
    new_prd_id = record.get("prd_id")
    
    if not message_id or not new_prd_id:
        raise ValueError(
            "Mensagem encontrada em agent_messages não contém id ou prd_id válido."
        )

    # Buscar PRD novo
    try:
        prd_response = (
            supabase.table("prd_documents")
            .select("prd_id, content")
            .eq("prd_id", new_prd_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar PRD novo com prd_id={new_prd_id}: {exc}") from exc

    if not prd_response.data:
        raise ValueError(f"PRD novo com prd_id={new_prd_id} não encontrado")

    prd_record = prd_response.data[0]
    content = prd_record.get("content") or {}
    prd_payload = content.get("content") if isinstance(content, dict) else content
    raw_output = content.get("raw_output") if isinstance(content, dict) else None
    
    # Formatar PRD para texto
    if isinstance(prd_payload, dict):
        try:
            prd_text = json.dumps(prd_payload, ensure_ascii=False, indent=2)
        except Exception:
            prd_text = str(prd_payload)
    elif isinstance(raw_output, str):
        prd_text = raw_output
    else:
        prd_text = str(prd_payload) if prd_payload else ""

    return prd_text, new_prd_id, message_id


def get_previous_prd_for_comparison(new_prd_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Busca o PRD anterior mais recente (antes do novo) para comparação.
    Busca apenas PRDs com o mesmo project_id do projeto atual.
    Retorna (prd_text, old_prd_id) ou (None, None) se não existir.
    """
    try:
        # Buscar PRD atual para pegar created_at e project_id
        current_response = (
            supabase.table("prd_documents")
            .select("created_at, project_id")
            .eq("prd_id", new_prd_id)
            .limit(1)
            .execute()
        )
        
        if not current_response.data:
            return None, None
        
        current_record = current_response.data[0]
        current_created_at = current_record.get("created_at")
        current_project_id = current_record.get("project_id")
        
        if not current_project_id:
            print("⚠️  PRD atual não possui project_id, usando PROJECT_ID constante")
            current_project_id = PROJECT_ID
        
        # Buscar PRD anterior mais recente com o mesmo project_id
        response = (
            supabase.table("prd_documents")
            .select("prd_id, content")
            .eq("project_id", current_project_id)
            .lt("created_at", current_created_at)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        print(f"⚠️  Erro ao buscar PRD anterior: {exc}")
        return None, None

    if not response.data:
        return None, None

    record = response.data[0]
    old_prd_id = record.get("prd_id")
    content = record.get("content") or {}
    prd_payload = content.get("content") if isinstance(content, dict) else content
    raw_output = content.get("raw_output") if isinstance(content, dict) else None
    
    if isinstance(prd_payload, dict):
        try:
            prd_text = json.dumps(prd_payload, ensure_ascii=False, indent=2)
        except Exception:
            prd_text = str(prd_payload)
    elif isinstance(raw_output, str):
        prd_text = raw_output
    else:
        prd_text = str(prd_payload) if prd_payload else None
    
    return prd_text, old_prd_id


def get_latest_codegen_artifacts() -> List[Dict[str, str]]:
    """
    Busca os artifacts do codegen mais recente do projeto.
    Retorna lista vazia se não existir código gerado.
    """
    try:
        response = (
            supabase.table("codegen_documents")
            .select("codegen_id, content, created_at")
            .eq("project_id", PROJECT_ID)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        print(f"⚠️  Erro ao buscar codegen mais recente: {exc}")
        return []

    if not response.data:
        return []

    record = response.data[0]
    content = record.get("content") or {}
    codegen_content = content.get("content") if isinstance(content, dict) else content
    
    artifacts = []
    if isinstance(codegen_content, dict):
        artifacts_list = codegen_content.get("artifacts", [])
        if isinstance(artifacts_list, list):
            artifacts = [
                {"path": a.get("path", ""), "content": a.get("content", "")}
                for a in artifacts_list
                if isinstance(a, dict) and a.get("path")
            ]
    
    return artifacts


def build_user_message(
    new_prd_text: str,
    old_prd_text: Optional[str],
    old_prd_id: Optional[str],
    current_code_artifacts: List[Dict[str, str]],
    base_prompt: str
) -> str:
    """Constrói mensagem do usuário para LLM com PRDs e código atual."""
    sections = []
    
    if base_prompt:
        sections.append(base_prompt.strip())
    
    sections.append("[PRD NOVO]")
    sections.append(new_prd_text.strip())
    
    if old_prd_text and old_prd_id:
        sections.append("\n[PRD ANTERIOR]")
        sections.append(old_prd_text.strip())
        sections.append(f"\n[NOTA] Este é um PRD subsequente. Compare as diferenças entre o PRD novo e o anterior.")
    else:
        sections.append("\n[PRD ANTERIOR]")
        sections.append("Nenhum PRD anterior encontrado. Este é o primeiro PRD do projeto.")
        sections.append("\n[NOTA] Como este é o primeiro ciclo, todos os arquivos necessários baseados no PRD devem ser listados em 'files_to_create'. Não inclua 'files_to_modify' ou 'files_to_delete'.")
    
    if current_code_artifacts:
        sections.append("\n[CÓDIGO ATUAL]")
        sections.append(f"Lista de {len(current_code_artifacts)} arquivos do código atual:")
        # Limitar para não exceder tokens - mostrar primeiros 20 arquivos
        for artifact in current_code_artifacts[:20]:
            path = artifact.get("path", "")
            content_preview = artifact.get("content", "")[:500]  # Primeiros 500 chars
            sections.append(f"\n--- Arquivo: {path} ---")
            sections.append(content_preview)
            if len(artifact.get("content", "")) > 500:
                sections.append("... (conteúdo truncado)")
        if len(current_code_artifacts) > 20:
            sections.append(f"\n... e mais {len(current_code_artifacts) - 20} arquivos")
    else:
        sections.append("\n[CÓDIGO ATUAL]")
        sections.append("Nenhum código gerado ainda. Este é o primeiro ciclo do projeto.")
        sections.append("\n[NOTA] Como não há código existente, todos os arquivos do PRD devem ser listados em 'files_to_create'.")
    
    return "\n".join(sections)


def _extract_code_fence(raw_str: str) -> str:
    """Extrai conteúdo de code fence se presente."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_str, re.IGNORECASE)
    return match.group(1).strip() if match else raw_str


def _generate_default_first_cycle_report() -> dict:
    """
    Gera relatório padrão para o primeiro ciclo (sem PRD anterior nem código existente).
    Não precisa chamar LLM, pois é óbvio que todos os arquivos do PRD precisam ser criados.
    """
    return {
        "impact_report": {
            "summary": {
                "is_first_cycle": True,
                "overall_impact": "high",
                "total_files_to_create": 0,  # Será determinado pelo scaffold baseado no PRD
                "total_files_to_modify": 0,
                "total_files_to_delete": 0,
                "schema_changes_required": True,  # Provavelmente precisa criar schema inicial
                "summary_text": "Primeiro ciclo do projeto. Todos os arquivos necessários baseados no PRD devem ser criados. Não há código existente para modificar ou deletar."
            },
            "files_to_create": [],  # Scaffold vai determinar baseado no PRD
            "files_to_modify": [],
            "files_to_delete": [],
            "schema_changes": {
                "tables_to_create": [],  # Schema agent vai determinar baseado no PRD
                "tables_to_modify": [],
                "columns_to_add": [],
                "migration_notes": "Primeiro ciclo - schema completo será criado baseado no PRD."
            },
            "dependencies": {
                "affected_components": [],
                "affected_routes": []
            }
        }
    }


def _parse_impact_report(raw_str: str) -> dict:
    """Parse do relatório de impacto retornado pela LLM."""
    try:
        # Tentar extrair JSON de code fence
        raw_str = _extract_code_fence(raw_str)
        parsed = json.loads(raw_str)
    except json.JSONDecodeError:
        raise ValueError(
            "Não foi possível converter a resposta da LLM em JSON válido para o relatório de impacto. "
            f"Trecho inicial: {raw_str[:400]}"
        )

    if not isinstance(parsed, dict):
        raise ValueError("Resposta do analyzer_agent não é um objeto JSON.")

    impact_report = parsed.get("impact_report") or {}
    
    if not isinstance(impact_report, dict):
        raise ValueError("Campo impact_report deve ser objeto.")

    # Validar estrutura básica
    summary = impact_report.get("summary", {})
    if not isinstance(summary, dict):
        impact_report["summary"] = {}
    
    # Garantir que is_first_cycle existe
    if "is_first_cycle" not in impact_report.get("summary", {}):
        # Tentar inferir se for primeiro ciclo baseado nos arrays vazios
        files_to_modify = impact_report.get("files_to_modify", [])
        files_to_delete = impact_report.get("files_to_delete", [])
        if not files_to_modify and not files_to_delete and impact_report.get("files_to_create"):
            impact_report.setdefault("summary", {})["is_first_cycle"] = True
        else:
            impact_report.setdefault("summary", {})["is_first_cycle"] = False

    return {
        "impact_report": impact_report,
    }


def call_llm(
    system_message: Optional[str],
    user_message: str,
    model: Optional[str],
    provider: Optional[str],
    system_revision: Optional[str] = None,
    max_tokens: int = 8000,
) -> dict:
    result = llm_call_llm(
        system_message=system_message,
        user_message=user_message,
        model=model,
        provider=provider,
        system_revision=system_revision,
        max_tokens=max_tokens,
        default_max_tokens=8000,
        default_temperature=0.1,
        agent_name="Analyzer Agent",
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
    raw_str = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)
    print(f"📦 Raw output (analyzer_agent): {len(raw_str)} caracteres")
    
    parsed = _parse_impact_report(raw_str)
    
    return {
        "content": parsed,
        "metadata": result["metadata"],
        "raw_output": raw_output,
    }


def save_to_analyzer_documents(
    result: dict,
    new_prd_id: Optional[str],
    old_prd_id: Optional[str] = None
) -> dict:
    if not new_prd_id:
        raise ValueError("new_prd_id inválido: é necessário para registrar analyzer_documents")

    content_jsonb = {
        "metadata": result.get("metadata"),
        "impact_report": result.get("content", {}).get("impact_report", {}),
        "raw_output": result.get("raw_output"),
        "new_prd_id": new_prd_id,
        "old_prd_id": old_prd_id,
    }

    is_using_service = supabase_service_key is not None and supabase_service_key.strip() != ""
    write_client = create_client(supabase_url, supabase_service_key) if is_using_service else supabase

    insert_payload = {
        "project_id": PROJECT_ID,
        "new_prd_id": new_prd_id,
        "old_prd_id": old_prd_id,
        "content": content_jsonb,
    }

    response = write_client.table("analyzer_documents").insert(insert_payload).execute()
    if not response.data:
        raise ValueError("Nenhum registro foi inserido na tabela analyzer_documents")

    analyzer_record = response.data[0]
    analyzer_id = analyzer_record.get("analyzer_id")

    try:
        write_client.table("agent_messages")\
            .insert({
                "project_id": PROJECT_ID,
                "from_agent": ANALYZER_AGENT_NAME,
                "to_agent": SCAFFOLD_AGENT_NAME,
                "status": "pending",
                "message_content": MESSAGE_CONTENT_CREATED,
                "prd_id": new_prd_id,
                "analyzer_id": analyzer_id,
            })\
            .execute()
        print("log agent_messages (analyzer_agent → scaffold_agent) registrado")
    except Exception as log_error:
        print(f"⚠️  Falha ao registrar mensagem em agent_messages: {log_error}")

    return analyzer_record


if __name__ == "__main__":
    base_user_msg = CONFIG_USER_MESSAGE
    if not base_user_msg:
        raise ValueError(
            "Defina user_message em system/analyzer_config.json ou forneça user_message explicitamente."
        )

    system_msg = CONFIG_SYSTEM_MESSAGE
    ai_model = CONFIG_AI_MODEL
    provider = CONFIG_PROVIDER
    system_rev: Optional[str] = None
    message_id: Optional[str] = None

    try:
        # Buscar novo PRD a partir de mensagem pendente
        new_prd_text, new_prd_id, message_id = get_new_prd_from_message()

        if not new_prd_id or not new_prd_text or not message_id:
            print("no pending messages")
            raise SystemExit(0)

        # Marcar mensagem como working
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

        # Buscar PRD anterior e código atual
        print("📥 Buscando PRD anterior e código atual...")
        old_prd_text, old_prd_id = get_previous_prd_for_comparison(new_prd_id)
        current_code_artifacts = get_latest_codegen_artifacts()
        
        is_first_cycle = old_prd_text is None and len(current_code_artifacts) == 0
        
        if old_prd_text:
            print(f"✅ PRD anterior encontrado para comparação (old_prd_id: {old_prd_id})")
        else:
            print("ℹ️  Nenhum PRD anterior encontrado (primeiro ciclo)")
        
        print(f"📦 Código atual: {len(current_code_artifacts)} arquivos encontrados")
        
        if is_first_cycle:
            print("🆕 Primeiro ciclo detectado - gerando relatório padrão sem chamar LLM")
            # Gerar relatório padrão para primeiro ciclo (sem chamar LLM)
            default_report = _generate_default_first_cycle_report()
            resultado = {
                "content": default_report,
                "metadata": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "agent_model": None,
                    "provider": None,
                    "system_revision": None,
                    "generated_by": "default_first_cycle"
                },
                "raw_output": json.dumps(default_report, ensure_ascii=False, indent=2)
            }
            print("✅ Relatório padrão gerado (sem chamada LLM)")
        else:
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
                raise ValueError("System message não encontrado. Configure em analyzer_config.json ou no Supabase.")

            if ai_model is None:
                ai_model = "gpt-4o"

            if provider is None:
                provider = "openai"

            print("Parâmetros efetivos da execução (analyzer_agent):")
            print(f" - model: {ai_model}")
            print(f" - provider: {provider}")
            print(f" - notes: {CONFIG_PARAMETERS.get('notes') or '<não informado>'}")

            max_tokens_value = CONFIG_MAX_TOKENS if isinstance(CONFIG_MAX_TOKENS, int) else 8000

            user_message = build_user_message(new_prd_text, old_prd_text, old_prd_id, current_code_artifacts, base_user_msg)

            print("\n🤖 Gerando relatório de impacto via LLM...")
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
        impact_report = resultado.get("content", {}).get("impact_report", {})
        summary = impact_report.get("summary", {})
        overall_impact = summary.get("overall_impact", "unknown")
        files_to_create = len(impact_report.get("files_to_create", []))
        files_to_modify = len(impact_report.get("files_to_modify", []))
        files_to_delete = len(impact_report.get("files_to_delete", []))
        is_first_cycle_reported = summary.get("is_first_cycle", False)

        print(f"\n📊 Resumo do impacto:")
        print(f"   - Impacto geral: {overall_impact}")
        print(f"   - Primeiro ciclo: {is_first_cycle_reported}")
        print(f"   - Arquivos a criar: {files_to_create}")
        print(f"   - Arquivos a modificar: {files_to_modify}")
        print(f"   - Arquivos a deletar: {files_to_delete}")

        # Salvar no banco
        print("\n💾 Salvando relatório no Supabase...")
        saved_record = save_to_analyzer_documents(resultado, new_prd_id, old_prd_id)
        analyzer_id = saved_record.get("analyzer_id")
        print(f"✅ Relatório salvo com sucesso: analyzer_id={analyzer_id}")

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
        print(f"❌ Erro na execução do analyzer_agent: {e}")
        if message_id:
            try:
                supabase.table("agent_messages")\
                    .update({
                        "status": "error",
                        "message_content": f"analyzer_error: {str(e)}",
                    })\
                    .eq("id", message_id)\
                    .eq("project_id", PROJECT_ID)\
                    .execute()
            except Exception as exc:
                print(f"⚠️  Falha ao marcar mensagem como error: {exc}")
        raise

