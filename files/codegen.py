import os
import json
import ast
import re
import base64
from copy import deepcopy
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
CONFIG_PATH = Path(__file__).parent.parent / "system" / "codegen_config.json"

load_dotenv(dotenv_path=ENV_PATH)


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"⚠️  Falha ao ler codegen_config.json: {exc}")
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


CODEGEN_CONFIG = _load_config()
(
    CONFIG_PARAMETERS,
    CONFIG_SYSTEM_MESSAGE,
    CONFIG_USER_MESSAGE,
    CONFIG_AI_MODEL,
    CONFIG_PROVIDER,
) = _extract_config_values(CODEGEN_CONFIG)

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
CODEGEN_AGENT_NAME = "codegen_agent"
TESTER_AGENT_NAME = "tester_agent"
USER_REPORT_AGENT_NAME = "user_report"
MESSAGE_CONTENT_CREATED = "codegen_created"

SCAFFOLD_AGENT_NAME = "scaffold_agent"
ANALYZER_AGENT_NAME = "analyzer_agent"

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


def parse_codegen_content(raw: Any) -> dict:
    if isinstance(raw, dict):
        parsed = raw
    else:
        if raw is None:
            raise ValueError("Resposta da LLM vazia para o Codegen Agent")
        raw_str = str(raw).strip()
        if not raw_str:
            raise ValueError("Resposta da LLM vazia para o Codegen Agent")

        raw_str = _extract_code_fence(raw_str)
        parsed = _parse_jsonish(raw_str)
        if parsed is None:
            raise ValueError(
                "Não foi possível converter a resposta da LLM em JSON válido. Trecho inicial: "
                + raw_str[:500]
            )

    if isinstance(parsed, list):
        normalized = {"artifacts": [_normalize_artifact_entry(item) for item in parsed]}
    elif isinstance(parsed, dict):
        data = deepcopy(parsed)

        if "codegen" in data:
            codegen_value = data["codegen"]
            if isinstance(codegen_value, str):
                inner = _parse_jsonish(_extract_code_fence(codegen_value.strip()))
                if isinstance(inner, dict):
                    codegen_value = inner
                else:
                    raise ValueError("Não foi possível normalizar a chave 'codegen' da resposta da LLM")
            elif not isinstance(codegen_value, dict):
                raise ValueError("A chave 'codegen' deve ser um objeto JSON")

            content_dict = deepcopy(codegen_value)
            for key, value in data.items():
                if key != "codegen" and key not in content_dict:
                    content_dict[key] = value
            normalized = content_dict
        else:
            normalized = data

        normalized = _normalize_artifact_lists(normalized)
    else:
        raise ValueError("Resposta da LLM não contém um objeto JSON válido")

    artifacts = normalized.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Nenhum artifact foi retornado pelo Codegen Agent")

    try:
        sanitized = json.loads(json.dumps(normalized))
    except (TypeError, ValueError) as exc:
        raise ValueError("Conteúdo do Codegen contém tipos não serializáveis") from exc

    return sanitized


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


def _collect_scaffold_paths(scaffold_payload: Any) -> list[str]:
    paths: list[str] = []

    def _collect(entry: Any):
        if isinstance(entry, dict):
            path = entry.get("path")
            if isinstance(path, str) and path.strip():
                paths.append(path.strip())
        elif isinstance(entry, list):
            for item in entry:
                _collect(item)

    if isinstance(scaffold_payload, dict):
        for value in scaffold_payload.values():
            _collect(value)
    elif isinstance(scaffold_payload, list):
        _collect(scaffold_payload)

    return sorted(dict.fromkeys(paths))


def _format_scaffold_payload(scaffold_payload: Any, raw_output: Optional[str]) -> str:
    sections: list[str] = []

    if isinstance(scaffold_payload, dict):
        metadata = {k: v for k, v in scaffold_payload.items() if k not in {"files_root", "files_app", "files_lib", "files_api", "files_test"}}
        if metadata:
            try:
                sections.append("[SCAFFOLD SUMMARY]")
                sections.append(json.dumps(metadata, ensure_ascii=False, indent=2))
            except (TypeError, ValueError):
                sections.append("[SCAFFOLD SUMMARY]")
                sections.append(str(metadata))

        group_keys = ["files_root", "files_app", "files_lib", "files_api", "files_test"]
        for key in group_keys:
            artifacts = scaffold_payload.get(key)
            if not isinstance(artifacts, list) or not artifacts:
                continue

            group_lines = [f"[{key.upper()}]"]
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
                group_lines.append(f"--- {path} ---")
                group_lines.append(fence_header)
                group_lines.append(content_text)
                group_lines.append("```")
                group_lines.append("")

            if group_lines and group_lines[-1] == "":
                group_lines.pop()

            if len(group_lines) > 1:
                sections.append("\n".join(group_lines))

    if sections:
        return "\n\n".join(sections)

    if isinstance(scaffold_payload, (dict, list)):
        try:
            return json.dumps(scaffold_payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            pass

    if isinstance(raw_output, str) and raw_output.strip():
        return raw_output.strip()

    return str(scaffold_payload) if scaffold_payload is not None else ""


# === Supabase helpers ===
def get_system_message() -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    try:
        response = (
            supabase.table("system_message")
            .select("content, system_revision, ai_id, updated_at")
            .eq("is_active", True)
            .eq("agent_type", "codegen_agent")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='codegen_agent'")
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


def get_impact_report(analyzer_id: Optional[str]) -> Optional[dict]:
    """
    Busca o relatório de impacto do analyzer.
    """
    if not analyzer_id:
        return None
    
    try:
        response = (
            supabase.table("analyzer_documents")
            .select("content")
            .eq("analyzer_id", analyzer_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        print(f"⚠️  Erro ao buscar relatório de impacto: {exc}")
        return None

    if not response.data:
        return None

    record = response.data[0]
    content = record.get("content") or {}
    impact_report = content.get("impact_report") if isinstance(content, dict) else None
    
    return impact_report


def get_scaffold_from_message() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], list[str], Optional[str], Optional[dict]]:
    """
    Busca a mensagem mais recente de schema_agent -> codegen_agent em agent_messages
    e retorna o scaffold correspondente junto com prd_id, paths esperados e relatório de impacto.
    """
    try:
        response = (
            supabase.table("agent_messages")
            .select("id, scaffold_id, prd_id, schema_id, analyzer_id, status")
            .eq("project_id", PROJECT_ID)
            .eq("from_agent", "schema_agent")
            .eq("to_agent", CODEGEN_AGENT_NAME)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar agent_messages para codegen_agent: {exc}") from exc

    if not response.data:
        # Nenhuma mensagem pendente para este agente
        return None, None, None, None, None, [], None, None

    msg_record = response.data[0]
    message_id = msg_record.get("id")
    scaffold_id = msg_record.get("scaffold_id")
    prd_id = msg_record.get("prd_id")
    schema_id = msg_record.get("schema_id")
    analyzer_id = msg_record.get("analyzer_id")

    if not message_id or not scaffold_id or not schema_id:
        raise ValueError(
            "Mensagem encontrada em agent_messages não contém id, scaffold_id ou schema_id válido. "
            "Verifique se o schema_agent está salvando esses campos corretamente."
        )

    # Buscar scaffold correspondente em scaffold_documents
    try:
        scaffold_response = (
            supabase.table("scaffold_documents")
            .select("scaffold_id, prd_id, content")
            .eq("scaffold_id", scaffold_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar scaffold com scaffold_id={scaffold_id}: {exc}") from exc

    if not scaffold_response.data:
        raise ValueError(
            f"Scaffold com scaffold_id={scaffold_id} não encontrado em scaffold_documents"
        )

    record = scaffold_response.data[0]
    # Caso o prd_id não venha da mensagem, usar o do registro
    prd_id = prd_id or record.get("prd_id")

    content = record.get("content") or {}
    scaffold_payload = content.get("content") if isinstance(content, dict) else content
    raw_output = content.get("raw_output") if isinstance(content, dict) else None
    text_payload = _format_scaffold_payload(scaffold_payload, raw_output)
    expected_paths = _collect_scaffold_paths(scaffold_payload)

    # Buscar relatório de impacto se analyzer_id estiver disponível
    impact_report = None
    if analyzer_id:
        impact_report = get_impact_report(analyzer_id)

    return text_payload, scaffold_id, prd_id, schema_id, message_id, expected_paths, analyzer_id, impact_report


def get_prd_text(prd_id: str) -> str:
    if not prd_id:
        raise ValueError("prd_id inválido para buscar PRD")

    try:
        response = (
            supabase.table("prd_documents")
            .select("content")
            .eq("prd_id", prd_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar PRD com prd_id={prd_id}: {exc}") from exc

    if not response.data:
        raise ValueError(f"PRD com prd_id={prd_id} não encontrado em prd_documents")

    record = response.data[0]
    content = record.get("content") or {}
    prd_payload = content.get("content") if isinstance(content, dict) else content
    raw_output = content.get("raw_output") if isinstance(content, dict) else None
    return _format_prd_payload(prd_payload, raw_output)


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


def get_tester_correction_message() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[dict], Optional[list]]:
    """
    Busca a mensagem mais recente de tester_agent -> codegen_agent OU user_report -> codegen_agent
    em agent_messages que indica necessidade de correção. Retorna os IDs necessários, o relatório do tester
    e a lista de arquivos com erro do message_content.
    """
    try:
        response = (
            supabase.table("agent_messages")
            .select("id, codegen_id, prd_id, scaffold_id, schema_id, tester_id, status, message_content, from_agent")
            .eq("project_id", PROJECT_ID)
            .in_("from_agent", [TESTER_AGENT_NAME, USER_REPORT_AGENT_NAME])
            .eq("to_agent", CODEGEN_AGENT_NAME)
            .eq("status", "pending")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar agent_messages: {exc}") from exc

    if not response.data:
        # Nenhuma mensagem pendente do tester ou user_report
        return None, None, None, None, None, None, None, None

    msg_record = response.data[0]
    message_id = msg_record.get("id")
    codegen_id = msg_record.get("codegen_id")
    prd_id = msg_record.get("prd_id")
    scaffold_id = msg_record.get("scaffold_id")
    schema_id = msg_record.get("schema_id")
    tester_id = msg_record.get("tester_id")
    message_content = msg_record.get("message_content")
    from_agent = msg_record.get("from_agent")

    if not message_id or not codegen_id:
        raise ValueError(
            "Mensagem encontrada não contém id ou codegen_id válido."
        )
    
    # Se for mensagem manual (user_report), tester_id deve existir (criado pelo script)
    # Mas vamos tornar mais flexível para compatibilidade
    if not tester_id:
        print("⚠️  Mensagem sem tester_id - tentando continuar sem relatório")
        tester_report = None
    else:
        # Buscar relatório do tester
        try:
            tester_response = (
                supabase.table("tester_documents")
                .select("content")
                .eq("tester_id", tester_id)
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise ValueError(f"Erro ao buscar relatório do tester com tester_id={tester_id}: {exc}") from exc

        if not tester_response.data:
            print(f"⚠️  Mensagem encontrada (tester_id={tester_id}), mas relatório não encontrado.")
            tester_report = None
        else:
            tester_record = tester_response.data[0]
            tester_content = tester_record.get("content") or {}
            tester_report = tester_content.get("report") if isinstance(tester_content, dict) else tester_content

    # Extrair files_with_errors do message_content se for JSON estruturado
    files_with_errors = None
    if message_content:
        try:
            if isinstance(message_content, str):
                content_parsed = json.loads(message_content)
            else:
                content_parsed = message_content
            
            if isinstance(content_parsed, dict) and content_parsed.get("type") == "correction_request":
                files_with_errors = content_parsed.get("files_with_errors", [])
                if files_with_errors:
                    print(f"📋 Arquivos com erro identificados no message_content: {len(files_with_errors)} arquivos")
        except (json.JSONDecodeError, TypeError):
            # message_content não é JSON, é string simples (sem erros)
            pass

    return message_id, codegen_id, prd_id, scaffold_id, schema_id, tester_id, tester_report, files_with_errors


def get_codegen_artifacts(codegen_id: str) -> List[Dict[str, str]]:
    """Busca os artifacts do codegen anterior que precisa ser corrigido."""
    try:
        response = (
            supabase.table("codegen_documents")
            .select("content")
            .eq("codegen_id", codegen_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise ValueError(f"Erro ao buscar codegen com codegen_id={codegen_id}: {exc}") from exc

    if not response.data:
        raise ValueError(f"Codegen com codegen_id={codegen_id} não encontrado")

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


def _validate_react_query_setup(artifacts: List[Dict[str, str]], generated_paths: List[str]) -> None:
    """
    Valida se QueryClientProvider está configurado quando há uso de React Query.
    Detecta uso de useMutation, useQuery ou useQueryClient e verifica se Providers está no layout.
    """
    if not artifacts:
        return
    
    # Detectar uso de React Query
    react_query_hooks = ["useMutation", "useQuery", "useQueryClient"]
    files_using_react_query = []
    
    for artifact in artifacts:
        path = artifact.get("path", "").strip()
        content = artifact.get("content", "")
        
        if not path or not content:
            continue
        
        # Verificar se o arquivo usa algum hook do React Query
        for hook in react_query_hooks:
            if hook in content:
                files_using_react_query.append(path)
                break
    
    if not files_using_react_query:
        # Nenhum arquivo usa React Query, não precisa validar
        return
    
    print(f"🔍 Detectado uso de React Query em {len(files_using_react_query)} arquivo(s)")
    
    # Verificar se app/providers.tsx existe
    providers_path = "app/providers.tsx"
    providers_exists = any(
        a.get("path", "").strip() == providers_path 
        for a in artifacts
    )
    
    # Verificar se app/layout.tsx usa Providers e obter conteúdo do providers.tsx
    layout_path = "app/layout.tsx"
    layout_uses_providers = False
    providers_content = None
    
    for artifact in artifacts:
        path = artifact.get("path", "").strip()
        if path == layout_path:
            content = artifact.get("content", "")
            # Verificar se importa Providers e usa no JSX
            if "Providers" in content and ("<Providers>" in content or "<Providers " in content):
                layout_uses_providers = True
        elif path == providers_path:
            providers_content = artifact.get("content", "")
    
    # Verificar se providers.tsx tem QueryClientProvider
    providers_has_query_client = False
    if providers_content:
        if "QueryClientProvider" in providers_content and "QueryClient" in providers_content:
            providers_has_query_client = True
    
    # Reportar problemas
    issues = []
    
    if not providers_exists:
        issues.append(f"❌ ERRO CRÍTICO: `{providers_path}` não existe, mas há uso de React Query em: {', '.join(files_using_react_query[:3])}")
        print(f"   ⚠️  Arquivos usando React Query: {files_using_react_query[:5]}")
    elif not providers_has_query_client:
        issues.append(f"❌ ERRO CRÍTICO: `{providers_path}` existe mas não contém QueryClientProvider")
    
    if not layout_uses_providers:
        issues.append(f"❌ ERRO CRÍTICO: `{layout_path}` não envolve children com <Providers>, mas há uso de React Query")
    
    if issues:
        print("\n🚨 VALIDAÇÃO FALHOU - QueryClientProvider não configurado corretamente:")
        for issue in issues:
            print(f"   {issue}")
        print("\n   ⚠️  Isso causará erro de runtime: 'No QueryClient set, use QueryClientProvider to set one'")
        print("   ⚠️  O código gerado precisa ser corrigido antes de ser usado.")
    else:
        print("✅ Validação: QueryClientProvider configurado corretamente")


def build_correction_message(
    artifacts: List[Dict[str, str]],
    tester_report: dict,
    prd_text: Optional[str],
    schema_text: Optional[str],
    base_prompt: str,
    files_with_errors: Optional[List[str]] = None,
    impact_report: Optional[dict] = None
) -> str:
    """Constrói mensagem de correção incluindo código atual e erros encontrados."""
    sections = []
    
    # Prompt base
    if base_prompt:
        sections.append(base_prompt.strip())
    
    # Seção de correção
    sections.append("[CORREÇÃO NECESSÁRIA]")
    sections.append("O código gerado anteriormente contém erros que precisam ser corrigidos. Abaixo está o relatório de validação com os problemas encontrados.")
    
    # Relatório do tester
    if tester_report:
        try:
            report_json = json.dumps(tester_report, ensure_ascii=False, indent=2)
            sections.append("[RELATÓRIO DE VALIDAÇÃO]")
            sections.append(report_json)
        except Exception:
            sections.append(str(tester_report))
    
    # Extrair arquivos com erro: usar lista fornecida ou extrair do relatório
    if files_with_errors is None:
        files_with_errors = []
    
    all_file_paths = []
    
    # Se não foi fornecida lista, extrair do relatório
    if not files_with_errors and tester_report and isinstance(tester_report, dict):
        report_files = tester_report.get("files", [])
        if isinstance(report_files, list):
            for file_info in report_files:
                if isinstance(file_info, dict):
                    file_path = file_info.get("file_path", "")
                    file_status = file_info.get("status", "")
                    if file_path:
                        all_file_paths.append(file_path)
                        if file_status == "error":
                            files_with_errors.append(file_path)
    else:
        # Se foi fornecida lista, também precisamos de all_file_paths do relatório
        if tester_report and isinstance(tester_report, dict):
            report_files = tester_report.get("files", [])
            if isinstance(report_files, list):
                for file_info in report_files:
                    if isinstance(file_info, dict):
                        file_path = file_info.get("file_path", "")
                        if file_path:
                            all_file_paths.append(file_path)
    
    # Criar dicionário de artifacts por path para busca rápida
    artifacts_by_path = {a.get("path", ""): a for a in artifacts if a.get("path")}
    
    # Código atual - passar apenas arquivos com erro
    sections.append("[CÓDIGO ATUAL - ARQUIVOS COM ERRO]")
    sections.append("Abaixo estão apenas os arquivos que contêm erros e precisam ser corrigidos:")
    
    for file_path in files_with_errors:
        artifact = artifacts_by_path.get(file_path)
        if artifact:
            content = artifact.get("content", "")
            sections.append(f"\n--- Arquivo: {file_path} ---")
            sections.append(content)
    
    # Lista de todos os arquivos (para referência)
    if all_file_paths:
        sections.append("\n[LISTA COMPLETA DE ARQUIVOS]")
        sections.append("O projeto contém os seguintes arquivos (você deve retornar TODOS eles):")
        for path in all_file_paths:
            status_marker = " [COM ERRO - CORRIGIR]" if path in files_with_errors else " [OK - MANTER]"
            sections.append(f"- {path}{status_marker}")
    
    # Contexto adicional (PRD e Schema)
    if prd_text:
        sections.append("\n[PRD]")
        sections.append(prd_text.strip())
    
    if schema_text:
        sections.append("\n[SCHEMA]")
        sections.append(schema_text.strip())
    
    # Incluir relatório de impacto se disponível (para contexto de mudanças)
    if impact_report:
        sections.append("\n[RELATÓRIO DE IMPACTO - CONTEXTO]")
        summary = impact_report.get("summary", {})
        is_first_cycle = summary.get("is_first_cycle", False)
        if is_first_cycle:
            sections.append("Este é o primeiro ciclo. Todos os arquivos devem ser criados.")
        else:
            files_to_create = len(impact_report.get("files_to_create", []))
            files_to_modify = len(impact_report.get("files_to_modify", []))
            sections.append(f"Arquivos impactados pelo PRD: {files_to_create} a criar, {files_to_modify} a modificar")
    
    sections.append("\n[INSTRUÇÕES CRÍTICAS]")
    sections.append("1. Corrija APENAS os arquivos listados acima que têm erros reportados.")
    sections.append("2. Para arquivos marcados como [OK - MANTER], você NÃO precisa incluí-los na resposta (eles serão mantidos automaticamente).")
    sections.append("3. Para arquivos marcados como [COM ERRO - CORRIGIR], você DEVE incluí-los corrigidos na resposta.")
    sections.append("4. 🚨 CRÍTICO - Criar Arquivos Dependentes: Analise cuidadosamente o relatório de validação. Se um arquivo com erro menciona que módulos/arquivos estão faltando (ex.: 'Módulo @/lib/validations/customer não existe', 'Arquivo app/api/customers/route.ts precisa ser criado', 'Import @/types/customer falhará'), você DEVE criar esses arquivos dependentes na mesma resposta. Não basta corrigir apenas o arquivo com erro - você deve criar TODOS os arquivos que estão faltando e que são necessários para resolver os erros. Inclua esses arquivos novos nos artifacts retornados.")
    sections.append("5. Aplique todas as correções sugeridas no relatório de validação, incluindo criar arquivos dependentes mencionados nas sugestões de correção.")
    sections.append("6. Retorne no formato JSON com artifacts: (a) arquivos corrigidos, (b) arquivos novos que precisam ser criados para resolver dependências faltando.")
    
    return "\n\n".join(sections)


def build_user_message(scaffold_text: str, prd_text: Optional[str], schema_text: Optional[str], base_prompt: str, expected_files: Optional[list[str]], impact_report: Optional[dict] = None) -> str:
    prompt = (base_prompt or "").strip()
    scaffold_section = scaffold_text.strip()

    sections = []
    if prompt:
        sections.append(prompt)
    if prd_text:
        sections.append("[PRD]")
        sections.append(prd_text.strip())

    if schema_text:
        sections.append("[SCHEMA]")
        sections.append(schema_text.strip())

    sections.append("[SCAFFOLD]")
    if scaffold_section:
        sections.append(scaffold_section)

    # Incluir relatório de impacto se disponível
    if impact_report:
        sections.append("\n[RELATÓRIO DE IMPACTO - CODEGEN]")
        summary = impact_report.get("summary", {})
        is_first_cycle = summary.get("is_first_cycle", False)
        
        if is_first_cycle:
            sections.append("Este é o primeiro ciclo. Gere código completo para todos os arquivos listados em files_to_create.")
        else:
            sections.append("🚨 CRÍTICO: Este NÃO é o primeiro ciclo. Você DEVE gerar código APENAS para os arquivos impactados listados abaixo.")
            sections.append("NÃO gere código para arquivos que não estão na lista de impactados.")
            sections.append("NÃO inclua arquivos de configuração a menos que estejam explicitamente listados.")
            
            files_to_create = impact_report.get("files_to_create", [])
            files_to_modify = impact_report.get("files_to_modify", [])
            files_to_delete = impact_report.get("files_to_delete", [])
            
            total_impacted = len(files_to_create) + len(files_to_modify)
            sections.append(f"\nTotal de arquivos impactados: {len(files_to_create)} criar + {len(files_to_modify)} modificar = {total_impacted} arquivos")
            sections.append(f"Arquivos a deletar: {len(files_to_delete)}")
            sections.append(f"\nVocê deve retornar EXATAMENTE {total_impacted} arquivos nos artifacts (apenas os listados abaixo).")
            
            if files_to_create:
                sections.append("\n[ARQUIVOS A CRIAR - GERE CÓDIGO PARA ESTES]")
                for file_info in files_to_create:
                    path = file_info.get("path", "")
                    reason = file_info.get("reason", "")
                    priority = file_info.get("priority", "")
                    sections.append(f"- {path}: {reason} (prioridade: {priority})")
            
            if files_to_modify:
                sections.append("\n[ARQUIVOS A MODIFICAR - GERE CÓDIGO PARA ESTES]")
                for file_info in files_to_modify:
                    path = file_info.get("path", "")
                    reason = file_info.get("reason", "")
                    priority = file_info.get("priority", "")
                    changes = file_info.get("changes", [])
                    sections.append(f"- {path}: {reason} (prioridade: {priority})")
                    if changes:
                        for change in changes:
                            sections.append(f"  * {change}")
            
            if files_to_delete:
                sections.append("\n[ARQUIVOS A DELETAR - NÃO GERE CÓDIGO PARA ESTES]")
                for file_info in files_to_delete:
                    path = file_info.get("path", "")
                    reason = file_info.get("reason", "")
                    sections.append(f"- {path}: {reason}")
            
            sections.append("\n[INSTRUÇÃO FINAL]")
            sections.append(f"Retorne APENAS os {total_impacted} arquivos impactados listados acima. NÃO inclua outros arquivos.")

    if expected_files:
        lines = ["[EXPECTED_FILES]", "Liste completa de arquivos que devem ser emitidos exatamente com o conteúdo final:"]
        for path in expected_files:
            lines.append(f"- {path}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# === Função principal de chamada LLM ===
def call_llm(
    system_message: Optional[str],
    user_message: str,
    model: Optional[str],
    provider: Optional[str],
    system_revision: Optional[str] = None,
    max_tokens: int = 2048,
    expected_file_paths: Optional[list[str]] = None,
    mode: str = "criar",
    original_artifacts: Optional[List[Dict[str, str]]] = None,
    impact_report: Optional[dict] = None,
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
        raise ValueError("System message não disponível para o Codegen Agent")

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

    # Log do modo de execução
    if mode == "corrigir":
        print("🔧 Chamando LLM para corrigir código")
    else:
        print("📝 Chamando LLM para criar código")
    
    result = llm_call_llm(
        system_message=system_message,
        user_message=user_message,
        model=model,
        provider=provider,
        system_revision=system_revision,
        max_tokens=max_tokens,
        default_max_tokens=2048,
        default_temperature=0,
        agent_name="Codegen Agent",
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

    raw_output_str = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)
    print(f"📦 Raw output recebido: {len(raw_output_str)} caracteres")

    normalized_content = parse_codegen_content(raw_output)
    corrected_artifacts = normalized_content.get("artifacts") or []
    corrected_count = len(corrected_artifacts) if isinstance(corrected_artifacts, list) else 0
    print(f"✅ Artifacts corrigidos retornados: {corrected_count}")

    # No modo correção, mesclar artifacts corrigidos com os originais que não foram corrigidos
    corrected_by_path = {}
    if mode == "corrigir" and original_artifacts:
        print("🔧 Mesclando arquivos corrigidos com arquivos originais não corrigidos...")
        # Criar dicionário dos artifacts corrigidos por path
        if isinstance(corrected_artifacts, list):
            for artifact in corrected_artifacts:
                path = artifact.get("path", "")
                if path:
                    corrected_by_path[path.strip()] = artifact
        
        # Mesclar: usar corrigidos quando disponíveis, senão usar originais
        merged_artifacts = []
        for original in original_artifacts:
            path = original.get("path", "").strip()
            if path in corrected_by_path:
                # Arquivo foi corrigido, usar a versão corrigida
                merged_artifacts.append(corrected_by_path[path])
            else:
                # Arquivo não foi corrigido, manter original
                merged_artifacts.append(original)
        
        # Adicionar qualquer artifact corrigido que não estava nos originais (caso raro)
        for path, corrected_artifact in corrected_by_path.items():
            if not any(a.get("path", "").strip() == path for a in original_artifacts):
                merged_artifacts.append(corrected_artifact)
        
        artifacts = merged_artifacts
        artifact_count = len(artifacts)
        print(f"✅ Total de artifacts após mesclagem: {artifact_count} (corrigidos: {len(corrected_by_path)}, mantidos: {artifact_count - len(corrected_by_path)})")
    else:
        artifacts = corrected_artifacts
        artifact_count = corrected_count

    generated_paths = []
    if isinstance(artifacts, list):
        for artifact in artifacts:
            path_value = artifact.get("path")
            if isinstance(path_value, str):
                generated_paths.append(path_value.strip())

    # Validação baseada no relatório de impacto se disponível
    # Nota: impact_report é passado como parâmetro da função call_llm
    if mode == "criar" and impact_report:
        summary = impact_report.get("summary", {})
        is_first_cycle = summary.get("is_first_cycle", False)
        
        if not is_first_cycle:
            # Validar que apenas arquivos impactados foram gerados
            files_to_create = impact_report.get("files_to_create", [])
            files_to_modify = impact_report.get("files_to_modify", [])
            
            expected_impacted_paths = set()
            for file_info in files_to_create:
                path = file_info.get("path", "").strip()
                if path:
                    expected_impacted_paths.add(path)
            for file_info in files_to_modify:
                path = file_info.get("path", "").strip()
                if path:
                    expected_impacted_paths.add(path)
            
            generated_paths_set = set(generated_paths)
            
            # Verificar se há arquivos não impactados
            unexpected_paths = generated_paths_set - expected_impacted_paths
            if unexpected_paths:
                print(f"⚠️  AVISO: Codegen gerou {len(unexpected_paths)} arquivos não impactados: {list(unexpected_paths)[:5]}")
                print(f"   Esperado apenas {len(expected_impacted_paths)} arquivos impactados")
            
            # Verificar se faltam arquivos impactados
            missing_impacted = expected_impacted_paths - generated_paths_set
            if missing_impacted:
                print(f"⚠️  AVISO: Faltam {len(missing_impacted)} arquivos impactados: {list(missing_impacted)[:5]}")
    
    # Validação crítica: QueryClientProvider quando há uso de React Query
    _validate_react_query_setup(artifacts, generated_paths)

    if expected_file_paths:
        missing_paths = [path for path in expected_file_paths if path not in set(generated_paths)]
        if missing_paths:
            raise ValueError(
                "Codegen não retornou todos os arquivos exigidos pelo scaffold. "
                f"Faltantes: {missing_paths}"
            )

    # Atualizar normalized_content com artifacts mesclados (se aplicável)
    corrected_files_list = []
    if mode == "corrigir" and original_artifacts:
        normalized_content["artifacts"] = artifacts
        # Extrair lista de arquivos corrigidos para passar ao tester
        corrected_files_list = list(corrected_by_path.keys()) if corrected_by_path else []

    content_estimate_source = raw_output_str
    codegen_tokens = len(content_estimate_source) // 4 if content_estimate_source else 0

    metadata = {
        "prompt_tokens": usage_info.get("prompt_tokens", 0),
        "completion_tokens": usage_info.get("completion_tokens", 0),
        "total_tokens": usage_info.get("total_tokens", 0),
        "agent_model": result["metadata"].get("agent_model", model),
        "provider": result["metadata"].get("provider", provider),
        "artifact_count": artifact_count,
        "codegen_tokens": codegen_tokens,
        "system_revision": system_revision or "",
        "expected_file_count": len(expected_file_paths) if expected_file_paths else 0,
        "corrected_files": corrected_files_list if corrected_files_list else None,
    }

    return {
        "content": normalized_content,
        "metadata": metadata,
        "raw_output": raw_output,
    }


# === Persistência ===
def save_to_codegen_documents(result: dict, scaffold_id: Optional[str], prd_id: Optional[str], schema_id: Optional[str], corrected_files: Optional[List[str]] = None, analyzer_id: Optional[str] = None) -> dict:
    if not scaffold_id:
        raise ValueError("scaffold_id inválido: é necessário para registrar codegen_documents")

    content_jsonb = {
        "metadata": result.get("metadata"),
        "content": result.get("content"),
        "raw_output": result.get("raw_output"),
        "scaffold_id": scaffold_id,
        "prd_id": prd_id,
        "schema_id": schema_id,
    }

    is_using_service = supabase_service_key is not None and supabase_service_key.strip() != ""
    write_client = create_client(supabase_url, supabase_service_key) if is_using_service else supabase

    insert_payload = {
        "project_id": PROJECT_ID,
        "scaffold_id": scaffold_id,
        "prd_id": prd_id,
        "schema_id": schema_id,
        "content": content_jsonb,
    }

    response = write_client.table("codegen_documents").insert(insert_payload).execute()

    if not response.data:
        raise ValueError("Nenhum registro foi inserido na tabela codegen_documents")

    codegen_record = response.data[0]
    codegen_id = codegen_record.get("codegen_id")

    # Criar message_content: JSON estruturado se houver arquivos corrigidos, string simples se não houver
    if corrected_files and len(corrected_files) > 0:
        message_content = json.dumps({
            "type": "test_request",
            "corrected_files": corrected_files
        }, ensure_ascii=False)
    else:
        message_content = MESSAGE_CONTENT_CREATED

    try:
        write_client.table("agent_messages")\
            .insert({
                "project_id": PROJECT_ID,
                "from_agent": CODEGEN_AGENT_NAME,
                "to_agent": TESTER_AGENT_NAME,
                "status": "pending",
                "message_content": message_content,
                "prd_id": prd_id,
                "scaffold_id": scaffold_id,
                "schema_id": schema_id,
                "codegen_id": codegen_id,
                "analyzer_id": analyzer_id,
            })\
            .execute()
        if corrected_files:
            print(f"log agent_messages registrado com {len(corrected_files)} arquivos corrigidos para teste")
        else:
            print("log agent_messages registrado")
    except Exception as log_error:
        print(f"⚠️  Falha ao registrar mensagem em agent_messages: {log_error}")

    return codegen_record


# === Integração GitHub ===
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


def push_codegen_to_github(repo, artifacts: List[Dict[str, str]], prd_id: Optional[str], scaffold_id: Optional[str], codegen_id: Optional[str]) -> bool:
    """
    Faz commit e push dos artifacts gerados para o repositório GitHub.
    
    Args:
        repo: Objeto do repositório GitHub
        artifacts: Lista de artifacts com 'path' e 'content'
        prd_id: ID do PRD relacionado
        scaffold_id: ID do scaffold relacionado
        codegen_id: ID do codegen gerado
    
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
        
        # Se repositório está vazio, vamos criar o primeiro commit com os artifacts diretamente
        # Caso contrário, vamos criar/atualizar arquivos normalmente
        
        # Montar mensagem de commit
        commit_message_parts = ["Codegen"]
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


if __name__ == "__main__":
    base_user_msg = CONFIG_USER_MESSAGE
    if not base_user_msg:
        raise ValueError(
            "Defina user_message em system/codegen_config.json ou forneça user_message explicitamente."
        )

    message_id: Optional[str] = None

    try:
        # Primeiro verificar se há mensagens do tester_agent pedindo correção
        tester_msg_id, codegen_id, prd_id, scaffold_id, schema_id, tester_id, tester_report, files_with_errors = get_tester_correction_message()
        
        execution_mode = "criar"
        scaffold_paths = []
        original_artifacts = None  # Para mesclagem no modo correção
        
        if tester_msg_id and codegen_id and tester_report:
            # Modo correção: há mensagem do tester pedindo correção
            print("🔧 Modo correção: mensagem do tester encontrada")
            message_id = tester_msg_id
            execution_mode = "corrigir"
            
            # Marcar mensagem como em processamento
            try:
                result = supabase.table("agent_messages")\
                    .update({"status": "working"})\
                    .eq("id", message_id)\
                    .eq("project_id", PROJECT_ID)\
                    .execute()
                if result.data:
                    print(f"✅ Mensagem {message_id} marcada como 'working' (modo correção)")
                else:
                    print(f"⚠️  Nenhuma linha atualizada ao marcar mensagem {message_id} como 'working'")
            except Exception as exc:
                print(f"❌ Falha ao marcar mensagem como working: {exc}")
                raise
            
            # Buscar código atual e contexto
            artifacts = get_codegen_artifacts(codegen_id)
            original_artifacts = artifacts  # Guardar para mesclagem
            prd_text = get_prd_text(prd_id) if prd_id else ""
            schema_text = get_schema_summary(schema_id) if schema_id else ""
            
            # Extrair paths esperados dos artifacts
            scaffold_paths = [a.get("path", "") for a in artifacts if a.get("path")]
            
            # Buscar relatório de impacto se disponível (para contexto)
            impact_report_correction = None
            if codegen_id:
                # Tentar buscar analyzer_id da mensagem original
                try:
                    msg_response = (
                        supabase.table("agent_messages")
                        .select("analyzer_id")
                        .eq("project_id", PROJECT_ID)
                        .eq("codegen_id", codegen_id)
                        .order("created_at", desc=True)
                        .limit(1)
                        .execute()
                    )
                    if msg_response.data:
                        analyzer_id_correction = msg_response.data[0].get("analyzer_id")
                        if analyzer_id_correction:
                            impact_report_correction = get_impact_report(analyzer_id_correction)
                except Exception:
                    pass
            
            # Construir mensagem de correção (usar files_with_errors do message_content se disponível)
            user_message = build_correction_message(artifacts, tester_report, prd_text, schema_text, base_user_msg, files_with_errors, impact_report_correction)
            
            if files_with_errors:
                print(f"📝 Código a corrigir: {len(files_with_errors)} arquivos com erro (de {len(artifacts)} total)")
            else:
                print(f"📝 Código a corrigir: {len(artifacts)} arquivos")
            print(f"   - codegen_id: {codegen_id}")
            print(f"   - tester_id: {tester_id}")
        else:
            # Modo criação: buscar mensagem do schema_agent
            print("📝 Modo criação: buscando mensagem do schema_agent")
            scaffold_text, scaffold_id, prd_id, schema_id, message_id, scaffold_paths, analyzer_id, impact_report = get_scaffold_from_message()

            if not scaffold_id or not scaffold_text or not schema_id or not message_id:
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
                    print(f"✅ Mensagem {message_id} marcada como 'working' (modo criação)")
                else:
                    print(f"⚠️  Nenhuma linha atualizada ao marcar mensagem {message_id} como 'working'")
            except Exception as exc:
                print(f"❌ Falha ao marcar mensagem como working: {exc}")
                raise

            prd_text = get_prd_text(prd_id) if prd_id else ""
            schema_text = get_schema_summary(schema_id) if schema_id else ""
            
            # Log do relatório de impacto
            if impact_report:
                summary = impact_report.get("summary", {})
                is_first_cycle = summary.get("is_first_cycle", False)
                files_to_create = len(impact_report.get("files_to_create", []))
                files_to_modify = len(impact_report.get("files_to_modify", []))
                print(f"📊 Relatório de impacto encontrado:")
                print(f"   - Primeiro ciclo: {is_first_cycle}")
                print(f"   - Arquivos a criar: {files_to_create}")
                print(f"   - Arquivos a modificar: {files_to_modify}")
            else:
                print("⚠️  Relatório de impacto não encontrado. Gerando código completo.")
            
            user_message = build_user_message(scaffold_text, prd_text, schema_text, base_user_msg, scaffold_paths, impact_report)

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
            raise ValueError("System message não encontrado. Configure em codegen_config.json ou no Supabase.")

        if ai_model is None:
            ai_model = "gpt-4o"

        if provider is None:
            provider = "openai"

        print("Parâmetros efetivos da execução:")
        print(f" - model: {ai_model}")
        print(f" - provider: {provider}")
        print(f" - notes: {CONFIG_PARAMETERS.get('notes') or '<não informado>'}")

        max_tokens_value = CONFIG_MAX_TOKENS if isinstance(CONFIG_MAX_TOKENS, int) else 4000

        resultado = call_llm(
            system_message=system_message,
            user_message=user_message,
            model=ai_model,
            provider=provider,
            system_revision=system_revision,
            max_tokens=max_tokens_value,
            expected_file_paths=scaffold_paths,
            mode=execution_mode,
            original_artifacts=original_artifacts,
            impact_report=impact_report if execution_mode == "criar" else None,
        )

        llm_meta = resultado["metadata"]
        total_tokens = llm_meta["total_tokens"]
        print(f"resposta LLM: {total_tokens} total tokens")

        # Extrair lista de arquivos corrigidos do metadata se disponível
        corrected_files = resultado.get("metadata", {}).get("corrected_files")
        # Usar analyzer_id do modo criação ou buscar do modo correção
        current_analyzer_id = None
        if execution_mode == "criar" and 'analyzer_id' in locals():
            current_analyzer_id = analyzer_id
        elif execution_mode == "corrigir" and codegen_id:
            # Tentar buscar analyzer_id da mensagem original
            try:
                msg_response = (
                    supabase.table("agent_messages")
                    .select("analyzer_id")
                    .eq("project_id", PROJECT_ID)
                    .eq("codegen_id", codegen_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if msg_response.data:
                    current_analyzer_id = msg_response.data[0].get("analyzer_id")
            except Exception:
                pass
        saved_record = save_to_codegen_documents(resultado, scaffold_id, prd_id, schema_id, corrected_files, current_analyzer_id)
        codegen_tokens = llm_meta.get("codegen_tokens")
        print(f"codegen salvo com sucesso: {codegen_tokens} tokens")
        codegen_id = saved_record.get('codegen_id')
        print(f"codegen_id: {codegen_id}")

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
        print(f"❌ Erro na execução do codegen_agent: {e}")
        if message_id:
            try:
                supabase.table("agent_messages")\
                    .update({
                        "status": "error",
                        "message_content": f"codegen_error: {str(e)}",
                    })\
                    .eq("id", message_id)\
                    .eq("project_id", PROJECT_ID)\
                    .execute()
            except Exception as exc:
                print(f"⚠️  Falha ao marcar mensagem como error: {exc}")
        raise

