import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

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

client = OpenAI(api_key=api_key)

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
if supabase_service_key:
    supabase_write: Client = create_client(supabase_url, supabase_service_key)
else:
    supabase_write = supabase


def get_system_message() -> str:
    """
    Busca o system message mais recente da tabela system_message do Supabase.
    
    Filtra por is_active=true e agent_type='prd_agent', ordena por updated_at DESC
    e extrai o valor de content.
    
    Returns:
        Texto do system message como string
        
    Raises:
        ValueError: Se nenhum registro for encontrado ou se o campo content estiver vazio
    """
    try:
        response = supabase.table("system_message")\
            .select("content, updated_at")\
            .eq("is_active", True)\
            .eq("agent_type", "prd_agent")\
            .order("updated_at", desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data or len(response.data) == 0:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='prd_agent'")
        
        record = response.data[0]
        content = record.get("content")
        
        if not content:
            raise ValueError("Campo 'content' está vazio no registro encontrado")
        
        # Se content for uma string, retorna diretamente
        # Se for um objeto/dict, converte para string
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Se for um objeto, converte para string JSON ou retorna vazio
            return str(content) if content else ""
        else:
            # Para outros tipos (list, etc), converte para string
            return str(content)
        
    except Exception as e:
        print(f"Erro ao buscar system message do Supabase: {e}")
        raise


def call_llm(system_message: str = None, user_message: str = None, model: str = "gpt-4"):
    """
    Chama a API da OpenAI com mensagens de sistema e usuário.
    
    Se system_message não for fornecido, busca automaticamente do Supabase.
    
    Args:
        system_message: Mensagem do sistema que define o comportamento do assistente (opcional, busca do Supabase se None)
        user_message: Mensagem do usuário
        model: Modelo a ser usado (padrão: gpt-4)
    
    Returns:
        dict: Dicionário contendo:
            - response: Resposta da LLM
            - metadata: JSON com meta (tokens e agent_model), user_message e system_message
    """
    try:
        # Se system_message não for fornecido, busca do Supabase
        if system_message is None:
            system_message = get_system_message()
        
        if not user_message:
            raise ValueError("user_message é obrigatório")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        
        # Extrai informações de uso (tokens)
        usage = response.usage
        total_tokens = usage.total_tokens if usage else 0
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        
        # Monta o JSON de metadados
        metadata = {
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "agent_model": model
            },
            "user_message": user_message,
            "system_message": system_message
        }
        
        # Retorna a resposta e os metadados
        return {
            "response": response.choices[0].message.content,
            "metadata": metadata
        }
    
    except Exception as e:
        print(f"Erro ao chamar a LLM: {e}")
        raise


def save_to_prd_documents(result: dict):
    """
    Salva o resultado da LLM (response + metadata) como JSONB na tabela prd_documents.
    
    Usa a chave anônima (SUPABASE_KEY) para escrita no banco.
    Requer que as políticas RLS estejam configuradas no Supabase para permitir inserções.
    
    Args:
        result: Dicionário contendo 'response' e 'metadata' retornado por call_llm()
    
    Returns:
        dict: Dados do registro inserido
        
    Raises:
        Exception: Se houver erro ao inserir no Supabase
    """
    try:
        # Monta o JSONB com response e metadata
        content_jsonb = {
            "response": result.get("response"),
            "metadata": result.get("metadata")
        }
        
        print(f"Tentando inserir na tabela prd_documents...")
        
        # Insere na tabela prd_documents usando o cliente de escrita (service role ou anon)
        response = supabase_write.table("prd_documents")\
            .insert({"content": content_jsonb})\
            .execute()
        
        if not response.data or len(response.data) == 0:
            raise ValueError("Nenhum registro foi inserido na tabela prd_documents")
        
        return response.data[0]
    
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
        print(f"\n🔍 Soluções:")
        if not supabase_service_key:
            print(f"   ✅ Configure SUPABASE_SERVICE_ROLE_KEY no arquivo .env")
            print(f"      (Service role key bypassa RLS e permite escrita)")
        else:
            print(f"   - Verifique se a service role key está correta")
            print(f"   - Verifique se não há constraints ou triggers bloqueando")
        
        raise


if __name__ == "__main__":
    # Mensagem do usuário literal no script
    user_msg = "gere um prd"
    
    # Informa qual chave está sendo usada para escrita
    if supabase_service_key:
        print("ℹ️  Usando SUPABASE_SERVICE_ROLE_KEY para escrita (bypassa RLS)")
    else:
        print("ℹ️  Usando SUPABASE_KEY (anon) para escrita")
    
    print("\nBuscando system_message do Supabase...")
    system_msg = get_system_message()
    print("System message encontrado!")
    
    print("\nChamando LLM...")
    resultado = call_llm(system_message=system_msg, user_message=user_msg)
    
    print("\nSalvando resultado na tabela prd_documents...")
    saved_record = save_to_prd_documents(resultado)
    
    print(f"\n✅ PRD gerado e salvo com sucesso!")
    print(f"prd_id: {saved_record.get('prd_id')}")
    print(f"\nResposta da LLM:")
    print(resultado['response'][:500] + "..." if len(resultado['response']) > 500 else resultado['response'])

