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
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    # Tenta ler diretamente do arquivo como fallback
    if env_path.exists():
        content = env_path.read_text(encoding='utf-8-sig')
        for line in content.strip().split('\n'):
            if line.startswith('SUPABASE_URL=') and not supabase_url:
                supabase_url = line.split('=', 1)[1].strip()
            elif line.startswith('SUPABASE_KEY=') and not supabase_key:
                supabase_key = line.split('=', 1)[1].strip()

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e/ou SUPABASE_KEY não encontradas no arquivo .env")

supabase: Client = create_client(supabase_url, supabase_key)


def get_system_message() -> str:
    """
    Busca o system message mais recente da tabela system_message do Supabase.
    
    Filtra por is_active=true e agent_type='scaffold_agent', ordena por updated_at DESC
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
            .eq("agent_type", "scaffold_agent")\
            .order("updated_at", desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data or len(response.data) == 0:
            raise ValueError("Nenhum system message ativo encontrado para agent_type='scaffold_agent'")
        
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


if __name__ == "__main__":
    # Exemplo de uso - system_message será buscado automaticamente do Supabase
    user_msg = "pode me dizer o que é python?"
    
    resultado = call_llm(user_message=user_msg)
    
    # Exibe a resposta
    print(f"Resposta: {resultado['response']}")
    
    # Exibe o JSON de metadados
    print("\n--- Metadados ---")
    print(json.dumps(resultado['metadata'], indent=2, ensure_ascii=False))
