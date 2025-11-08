import os
import json
import ast
import re
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from typing import Optional, Tuple, Any

# Carrega variáveis de ambiente do arquivo .env na raiz do projeto
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Inicializa o cliente OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Tenta ler diretamente do arquivo como fallback
    if env_path.exists():
        content = env_path.read_text(encoding='utf-8-sig')  # utf-8-sig remove BOM
        for line in content.strip().split('\n'):
            if line.startswith('OPENAI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                break

    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")

openai_client = OpenAI(api_key=api_key)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_api_key:
    try:
        from anthropic import Anthropic  # type: ignore
        anthropic_client = Anthropic(api_key=anthropic_api_key)
    except ImportError:  # pragma: no cover - pacote pode não estar instalado
        anthropic_client = None
        print("⚠️  Biblioteca 'anthropic' não encontrada; defina o provider como openai ou instale o pacote.")
else:
    anthropic_client = None

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

    if not isinstance(parsed, dict):
        raise ValueError("Resposta da LLM não contém um objeto JSON válido")

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
    
    try:
        fetched_message = fetched_revision = fetched_model = fetched_provider = None
        if (
            system_message is None
            or system_revision is None
            or model is None
            or (isinstance(model, str) and model.strip() == "")
            or provider is None
            or (isinstance(provider, str) and provider.strip() == "")
        ):
            fetched_message, fetched_revision, fetched_model, fetched_provider = get_system_message()
        
        if system_message is None:
            system_message = fetched_message
        if system_revision is None:
            system_revision = fetched_revision
        if model is None or (isinstance(model, str) and model.strip() == ""):
            model = fetched_model or "gpt-4o"
        if provider is None or (isinstance(provider, str) and provider.strip() == ""):
            provider = fetched_provider
        if provider is None or provider.strip() == "":
            if isinstance(model, str) and model.lower().startswith("claude"):
                provider = "anthropic"
            else:
                provider = "openai"
        provider = provider.lower()

        if provider == "openai" and isinstance(model, str) and model.lower().startswith("claude"):
            print("⚠️  Modelo claude solicitado com provider openai; usando gpt-4o por padrão.")
            model = "gpt-4o"
        
        if not user_message:
            raise ValueError("user_message é obrigatório")
        
        print("chamando LLM")
        prd_content_raw: Optional[str] = None
        usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        if provider == "anthropic":
            if not anthropic_client:
                raise ValueError("Anthropic não configurado. Defina ANTHROPIC_API_KEY ou ajuste o provider.")
            response = anthropic_client.messages.create(
                model=model,
                system=system_message,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_message}
                    ]
                }]
            )
            prd_content_raw = "".join(
                part.text for part in response.content if hasattr(part, "text")
            )
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "input_tokens", 0)
                completion_tokens = getattr(usage, "output_tokens", 0)
                usage_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
        else:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                max_tokens=max_tokens
            )
            prd_content_raw = response.choices[0].message.content
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)
            usage_info = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        
        if prd_content_raw is None:
            raise ValueError("Resposta da LLM vazia")
        
        # Normaliza o conteúdo antes de prosseguir
        normalized_content = parse_prd_content(prd_content_raw)
        
        # Estima o tamanho do PRD em tokens (aproximação: 1 token ≈ 4 caracteres)
        content_estimate_source = prd_content_raw if isinstance(prd_content_raw, str) else json.dumps(prd_content_raw)
        prd_tokens = len(content_estimate_source) // 4 if content_estimate_source else 0
        
        # Monta o JSON de metadados
        metadata = {
            "prompt_tokens": usage_info["prompt_tokens"],
            "completion_tokens": usage_info["completion_tokens"],
            "total_tokens": usage_info["total_tokens"],
            "agent_model": model,
            "provider": provider,
            "system_revision": system_revision or "",
            "prd_tokens": prd_tokens
        }
        
        # Retorna a resposta e os metadados
        return {
            "content": normalized_content,
            "metadata": metadata
        }
    
    except Exception as e:
        print(f"Erro ao chamar a LLM: {e}")
        raise


def save_to_prd_documents(result: dict):
    try:
        # Monta o JSONB com content normalizado e metadata
        content_jsonb = {
            "metadata": result.get("metadata"),
            "content": result.get("content")
        }

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

        try:
            write_client.table("agent_messages")\
                .insert({
                    "project_id": PROJECT_ID,
                    "from_agent": PRD_AGENT_NAME,
                    "to_agent": SCAFFOLD_AGENT_NAME,
                    "status": MESSAGE_CONTENT_CREATED,
                    "message_content": MESSAGE_CONTENT_CREATED,
                    "prd_id": prd_id
                })\
                .execute()
            print("log agent_messages registrado")
        except Exception as log_error:
            print(f"⚠️  Falha ao registrar mensagem em agent_messages: {log_error}")

        return prd_record

    except Exception as e:
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
    # Mensagem do usuário literal no script
    user_msg = "Quero um sistema para cadastro de clientes com formulário de nome, e-mail e telefone. Preciso de uma tabela de listagem com busca e filtros, e um dashboard inicial com número total de clientes e um gráfico simples de crescimento mensal.Gostaria de ter modo escuro, autenticação simples por e-mail e uma página de configurações do usuário."

    # Busca system_message do Supabase
    system_msg, system_rev, system_model, system_provider = get_system_message()
    if system_rev:
        print(f"system_revision: {system_rev}")
    if system_model:
        print(f"system_model: {system_model}")
    if system_provider:
        print(f"system_provider: {system_provider}")
    
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

