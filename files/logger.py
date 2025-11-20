"""
Módulo centralizado de logging para todos os agentes.
Permite rastrear todos os passos de execução com controle granular via configuração.
"""
import os
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# === Caminhos e configuração ===
ENV_PATH = Path(__file__).parent.parent / ".env"
LOCAL_ENV_PATH = Path(__file__).parent.parent / "local.env"

load_dotenv(dotenv_path=ENV_PATH)
if LOCAL_ENV_PATH.exists():
    load_dotenv(dotenv_path=LOCAL_ENV_PATH)

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
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem estar definidos no .env")

supabase: Client = create_client(supabase_url, supabase_key)

PROJECT_ID = "639e810b-9d8c-4f31-9569-ecf61fb43888"

# Níveis de log em ordem de severidade
LOG_LEVELS = {
    "debug": 0,
    "info": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}

# Estado global do logger
_logger_state = {
    "agent_name": None,
    "enabled": False,
    "min_level": "info",
}


def init_logger(agent_name: str, config: dict) -> None:
    """
    Inicializa o logger com configuração do agente.
    
    Args:
        agent_name: Nome do agente (ex: 'prd_agent', 'analyzer_agent')
        config: Dicionário de configuração do agente (geralmente CONFIG_PARAMETERS)
    """
    _logger_state["agent_name"] = agent_name
    _logger_state["enabled"] = config.get("enable_logging", False)
    _logger_state["min_level"] = config.get("log_level", "info").lower()
    
    if _logger_state["enabled"]:
        log_info("agent_start", f"Logger inicializado para {agent_name}", details={
            "log_level": _logger_state["min_level"]
        })


def _should_log(level: str) -> bool:
    """Verifica se deve logar baseado na configuração."""
    if not _logger_state["enabled"]:
        return False
    
    min_level_value = LOG_LEVELS.get(_logger_state["min_level"], 1)
    level_value = LOG_LEVELS.get(level.lower(), 1)
    
    return level_value >= min_level_value


def log(
    level: str,
    log_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    raw_data: Optional[str] = None,
    prd_id: Optional[str] = None,
    analyzer_id: Optional[str] = None,
    scaffold_id: Optional[str] = None,
    schema_id: Optional[str] = None,
    codegen_id: Optional[str] = None,
    tester_id: Optional[str] = None,
    message_id: Optional[str] = None,
    execution_time_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
    model_used: Optional[str] = None,
    provider_used: Optional[str] = None,
    stack_trace: Optional[str] = None,
) -> Optional[str]:
    """
    Função principal de logging.
    
    Args:
        level: Nível do log ('debug', 'info', 'warning', 'error', 'critical')
        log_type: Tipo de evento ('llm_call', 'parse_error', 'save_document', etc)
        message: Mensagem descritiva
        details: Dados estruturados opcionais (JSONB)
        raw_data: Dados brutos para debug (respostas LLM, stack traces)
        **kwargs: Relacionamentos opcionais e metadados
    
    Returns:
        log_id se sucesso, None caso contrário
    """
    if not _should_log(level):
        return None
    
    if not _logger_state["agent_name"]:
        print("⚠️  Logger não inicializado. Chame init_logger() primeiro.")
        return None
    
    # Preparar payload
    payload = {
        "project_id": PROJECT_ID,
        "agent_name": _logger_state["agent_name"],
        "log_level": level.lower(),
        "log_type": log_type,
        "message": message,
    }
    
    if details:
        payload["details"] = details
    
    if raw_data:
        # Limitar tamanho do raw_data para evitar problemas (1MB)
        if len(raw_data) > 1000000:
            payload["raw_data"] = raw_data[:1000000] + f"\n... (truncado, tamanho original: {len(raw_data)} chars)"
        else:
            payload["raw_data"] = raw_data
    
    # Relacionamentos opcionais
    if prd_id:
        payload["prd_id"] = prd_id
    if analyzer_id:
        payload["analyzer_id"] = analyzer_id
    if scaffold_id:
        payload["scaffold_id"] = scaffold_id
    if schema_id:
        payload["schema_id"] = schema_id
    if codegen_id:
        payload["codegen_id"] = codegen_id
    if tester_id:
        payload["tester_id"] = tester_id
    if message_id:
        payload["message_id"] = message_id
    
    # Metadados
    if execution_time_ms is not None:
        payload["execution_time_ms"] = execution_time_ms
    if tokens_used is not None:
        payload["tokens_used"] = tokens_used
    if model_used:
        payload["model_used"] = model_used
    if provider_used:
        payload["provider_used"] = provider_used
    if stack_trace:
        payload["stack_trace"] = stack_trace
    
    # Tentar salvar no banco
    try:
        is_using_service = supabase_service_key is not None and supabase_service_key.strip() != ""
        write_client = create_client(supabase_url, supabase_service_key) if is_using_service else supabase
        
        response = write_client.table("agent_logs").insert(payload).execute()
        
        if response.data:
            log_id = response.data[0].get("log_id")
            return log_id
    except Exception as e:
        # Não falhar se logging falhar, apenas avisar
        print(f"⚠️  Erro ao salvar log: {e}")
    
    return None


def log_debug(log_type: str, message: str, **kwargs) -> Optional[str]:
    """Helper para log de nível debug."""
    return log("debug", log_type, message, **kwargs)


def log_info(log_type: str, message: str, **kwargs) -> Optional[str]:
    """Helper para log de nível info."""
    return log("info", log_type, message, **kwargs)


def log_warning(log_type: str, message: str, **kwargs) -> Optional[str]:
    """Helper para log de nível warning."""
    return log("warning", log_type, message, **kwargs)


def log_error(log_type: str, message: str, **kwargs) -> Optional[str]:
    """Helper para log de nível error."""
    return log("error", log_type, message, **kwargs)


def log_critical(log_type: str, message: str, **kwargs) -> Optional[str]:
    """Helper para log de nível critical."""
    return log("critical", log_type, message, **kwargs)


def log_llm_call(
    message: str,
    model: str,
    provider: str,
    execution_time_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
    raw_response: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Wrapper específico para logar chamadas LLM.
    
    Args:
        message: Descrição da chamada
        model: Modelo usado
        provider: Provider usado
        execution_time_ms: Tempo de execução
        tokens_used: Tokens consumidos
        raw_response: Resposta bruta da LLM (opcional)
        **kwargs: Outros parâmetros (prd_id, codegen_id, etc)
    """
    details = {
        "model": model,
        "provider": provider,
    }
    if execution_time_ms:
        details["execution_time_ms"] = execution_time_ms
    if tokens_used:
        details["tokens_used"] = tokens_used
    
    return log_info(
        "llm_call",
        message,
        details=details,
        raw_data=raw_response,
        model_used=model,
        provider_used=provider,
        execution_time_ms=execution_time_ms,
        tokens_used=tokens_used,
        **kwargs
    )


def log_llm_response(
    message: str,
    model: str,
    provider: str,
    tokens_used: Optional[int] = None,
    raw_response: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Wrapper para logar respostas LLM.
    """
    return log_info(
        "llm_response",
        message,
        details={"model": model, "provider": provider},
        raw_data=raw_response,
        model_used=model,
        provider_used=provider,
        tokens_used=tokens_used,
        **kwargs
    )


def log_parse_error(
    message: str,
    error_message: str,
    raw_data: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Wrapper para logar erros de parsing.
    
    Args:
        message: Mensagem descritiva
        error_message: Mensagem de erro do parser
        raw_data: Dados brutos que falharam no parsing
        **kwargs: Outros parâmetros
    """
    return log_error(
        "parse_error",
        message,
        details={"error_message": error_message},
        raw_data=raw_data,
        **kwargs
    )


def log_parse_success(
    message: str,
    parsed_size: Optional[int] = None,
    **kwargs
) -> Optional[str]:
    """Wrapper para logar parsing bem-sucedido."""
    details = {}
    if parsed_size:
        details["parsed_size"] = parsed_size
    
    return log_debug(
        "parse_success",
        message,
        details=details if details else None,
        **kwargs
    )


def log_agent_start(message: str = "Iniciando execução do agente", **kwargs) -> Optional[str]:
    """Log de início de execução do agente."""
    return log_info("agent_start", message, **kwargs)


def log_agent_end(
    message: str = "Execução do agente concluída",
    execution_time_ms: Optional[int] = None,
    **kwargs
) -> Optional[str]:
    """Log de fim de execução do agente."""
    return log_info(
        "agent_end",
        message,
        execution_time_ms=execution_time_ms,
        **kwargs
    )


def log_exception(
    log_type: str,
    message: str,
    exception: Exception,
    **kwargs
) -> Optional[str]:
    """
    Wrapper para logar exceções com stack trace.
    
    Args:
        log_type: Tipo de log
        message: Mensagem descritiva
        exception: Exceção capturada
        **kwargs: Outros parâmetros
    """
    stack_trace = traceback.format_exc()
    error_details = {
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
    }
    
    return log_error(
        log_type,
        message,
        details=error_details,
        stack_trace=stack_trace,
        **kwargs
    )


def log_save_document(
    message: str,
    document_type: str,
    document_id: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Wrapper para logar salvamento de documentos.
    
    Args:
        message: Mensagem descritiva
        document_type: Tipo de documento ('prd', 'analyzer', 'scaffold', etc)
        document_id: ID do documento salvo
        **kwargs: Outros parâmetros (prd_id, codegen_id, etc)
    """
    details = {"document_type": document_type}
    if document_id:
        details["document_id"] = document_id
    
    return log_info(
        "save_document",
        message,
        details=details,
        **kwargs
    )

