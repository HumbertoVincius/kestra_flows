"""
Módulo centralizado para acesso a LLMs (OpenAI, Anthropic, Gemini).
"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


# === Caminhos e configuração ===
ENV_PATH = Path(__file__).parent.parent / ".env"
LOCAL_ENV_PATH = Path(__file__).parent.parent / "local.env"

load_dotenv(dotenv_path=ENV_PATH)
if LOCAL_ENV_PATH.exists():
    load_dotenv(dotenv_path=LOCAL_ENV_PATH)


def _load_env_var(var_name: str) -> Optional[str]:
    """
    Carrega uma variável de ambiente, tentando primeiro os.getenv,
    depois lendo diretamente do arquivo .env ou local.env.
    """
    value = os.getenv(var_name)
    if value:
        return value
    
    # Tenta ler do .env primeiro, depois local.env
    for env_file in [ENV_PATH, LOCAL_ENV_PATH]:
        if env_file.exists():
            try:
                env_content = env_file.read_text(encoding="utf-8-sig")
                for line in env_content.strip().split("\n"):
                    if line.startswith(f"{var_name}="):
                        value = line.split("=", 1)[1].strip()
                        if value:
                            return value
            except Exception:
                continue
    
    return None


def _init_clients() -> Tuple[OpenAI, Optional[Any], Optional[Any]]:
    """
    Inicializa os clientes OpenAI, Anthropic e Gemini.
    Retorna (openai_client, anthropic_client, gemini_client).
    """
    # OpenAI
    openai_api_key = _load_env_var("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env ou local.env")
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Anthropic
    anthropic_api_key = _load_env_var("ANTHROPIC_API_KEY")
    anthropic_client = None
    if anthropic_api_key:
        try:
            from anthropic import Anthropic
            anthropic_client = Anthropic(api_key=anthropic_api_key)
        except ImportError:
            print("⚠️  Biblioteca 'anthropic' não encontrada; defina provider como openai ou instale o pacote.")
    
    # Gemini
    gemini_api_key = _load_env_var("GEMINI_API_KEY")
    gemini_client = None
    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            gemini_client = genai
        except ImportError:
            print("⚠️  Biblioteca 'google-generativeai' não encontrada; instale o pacote para usar Gemini.")
    
    return openai_client, anthropic_client, gemini_client


# Inicializa os clientes globalmente
openai_client, anthropic_client, gemini_client = _init_clients()


def call_llm(
    system_message: Optional[str],
    user_message: str,
    model: Optional[str],
    provider: Optional[str],
    system_revision: Optional[str] = None,
    max_tokens: int = 8000,
    default_max_tokens: int = 8000,
    default_temperature: float = 0.1,
    agent_name: str = "LLM",
    get_system_message_fn: Optional[Callable[[], Tuple[str, Optional[str], Optional[str], Optional[str]]]] = None,
    config_ai_model: Optional[str] = None,
    config_provider: Optional[str] = None,
    config_max_tokens: Optional[int] = None,
    config_temperature: Optional[float] = None,
    config_top_p: Optional[float] = None,
    config_frequency_penalty: Optional[float] = None,
    config_presence_penalty: Optional[float] = None,
    config_stop: Optional[list] = None,
) -> dict:
    """
    Função unificada para chamar LLMs (OpenAI, Anthropic, Gemini).
    
    Args:
        system_message: Mensagem do sistema
        user_message: Mensagem do usuário (obrigatória)
        model: Nome do modelo a usar
        provider: Provedor ('openai', 'anthropic', 'gemini')
        system_revision: Revisão da mensagem do sistema
        max_tokens: Número máximo de tokens (pode ser sobrescrito por config_max_tokens)
        default_max_tokens: Valor padrão de max_tokens se não especificado
        default_temperature: Valor padrão de temperature se não especificado
        agent_name: Nome do agente para mensagens de log/erro
        get_system_message_fn: Função callback para buscar system message do Supabase
        config_ai_model: Modelo padrão da configuração
        config_provider: Provedor padrão da configuração
        config_max_tokens: Max tokens da configuração (sobrescreve max_tokens)
        config_temperature: Temperature da configuração (sobrescreve default_temperature)
        config_top_p: Top-p da configuração
        config_frequency_penalty: Frequency penalty da configuração
        config_presence_penalty: Presence penalty da configuração
        config_stop: Stop sequences da configuração
    
    Returns:
        dict com 'content', 'metadata' e 'raw_output'
    """
    # Normaliza strings
    if isinstance(model, str):
        model = model.strip() or None
    if isinstance(provider, str):
        provider = provider.strip().lower() or None
    
    # Busca system message se necessário
    need_lookup = system_message is None or model is None or provider is None
    
    fetched_message = fetched_revision = fetched_model = fetched_provider = None
    if need_lookup and get_system_message_fn:
        fetched_message, fetched_revision, fetched_model, fetched_provider = get_system_message_fn()
        if system_message is None:
            system_message = fetched_message
        if system_revision is None:
            system_revision = fetched_revision
        if model is None:
            model = fetched_model
        if provider is None:
            provider = fetched_provider
    
    # Validações
    if system_message is None:
        raise ValueError(f"System message não disponível para o {agent_name}")
    
    if model is None:
        model = config_ai_model or "gpt-4o"
    
    if provider is None:
        if isinstance(model, str) and model.lower().startswith("claude"):
            provider = "anthropic"
        elif isinstance(model, str) and ("gemini" in model.lower() or "gpt" not in model.lower() and "claude" not in model.lower()):
            provider = "gemini"
        else:
            provider = config_provider or "openai"
    provider = provider.lower()
    
    # Validação de incompatibilidade
    if provider == "openai" and isinstance(model, str) and model.lower().startswith("claude"):
        print("⚠️  Modelo claude solicitado com provider openai; usando gpt-4o por padrão.")
        model = "gpt-4o"
    
    if not user_message:
        raise ValueError("user_message é obrigatório")
    
    # Configura max_tokens
    configured_max_tokens = max_tokens
    if isinstance(config_max_tokens, int):
        configured_max_tokens = config_max_tokens
    elif configured_max_tokens is None:
        configured_max_tokens = default_max_tokens
    
    # Configura temperature
    temperature_value = default_temperature
    if config_temperature is not None:
        temperature_value = config_temperature
    
    print(f"chamando LLM ({agent_name.lower()})")
    raw_output: Optional[str] = None
    usage_info = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    
    # Chama o provider apropriado
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
    
    elif provider == "gemini":
        if not gemini_client:
            raise ValueError("Gemini não configurado. Defina GEMINI_API_KEY ou ajuste o provider.")
        import google.generativeai as genai
        
        # Prepara o modelo Gemini
        gemini_model_name = model if model else "gemini-1.5-pro"
        gemini_model = genai.GenerativeModel(gemini_model_name)
        
        # Combina system message e user message para Gemini
        # Gemini não tem system message separado, então combinamos
        full_prompt = f"{system_message}\n\n{user_message}" if system_message else user_message
        
        # Configurações de geração
        generation_config = {
            "max_output_tokens": configured_max_tokens,
            "temperature": temperature_value,
        }
        if config_top_p is not None:
            generation_config["top_p"] = config_top_p
        
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        raw_output = response.text if hasattr(response, "text") else str(response)
        
        # Gemini não fornece usage info diretamente na resposta padrão
        # Mas podemos estimar ou tentar acessar se disponível
        usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        if hasattr(response, "usage_metadata"):
            usage_metadata = response.usage_metadata
            usage_info = {
                "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(usage_metadata, "completion_token_count", 0),
                "total_tokens": getattr(usage_metadata, "total_token_count", 0),
            }
    
    else:  # OpenAI (padrão)
        openai_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature_value,
            "max_tokens": configured_max_tokens,
        }
        if config_top_p is not None:
            openai_kwargs["top_p"] = config_top_p
        if config_frequency_penalty is not None:
            openai_kwargs["frequency_penalty"] = config_frequency_penalty
        if config_presence_penalty is not None:
            openai_kwargs["presence_penalty"] = config_presence_penalty
        if config_stop is not None:
            openai_kwargs["stop"] = config_stop
        
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
        raise ValueError(f"Resposta da LLM vazia para o {agent_name}")
    
    raw_str = raw_output if isinstance(raw_output, str) else json.dumps(raw_output, ensure_ascii=False)
    print(f"📦 Raw output ({agent_name.lower()}): {len(raw_str)} caracteres")
    
    metadata = {
        "prompt_tokens": usage_info["prompt_tokens"],
        "completion_tokens": usage_info["completion_tokens"],
        "total_tokens": usage_info["total_tokens"],
        "agent_model": model,
        "provider": provider,
        "system_revision": system_revision or "",
    }
    
    return {
        "content": raw_output,
        "metadata": metadata,
        "raw_output": raw_output,
    }

