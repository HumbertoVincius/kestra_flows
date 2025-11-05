# Arquivo: files/scaffold.py

import os
from openai import OpenAI

# Inicializa o cliente OpenAI
# A API key deve estar configurada na variável de ambiente OPENAI_API_KEY
# ou você pode passar diretamente: OpenAI(api_key="sua-api-key-aqui")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System message - define o comportamento e contexto do assistente
system_message = """
Você é um assistente útil e prestativo.
"""

# Prompt message - a mensagem do usuário
prompt_message = """
Digite sua mensagem aqui.
"""

# Faz a chamada para a OpenAI
response = client.chat.completions.create(
    model="gpt-4o-mini",  # ou outro modelo disponível
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_message}
    ],
    temperature=0.7,
    max_tokens=1000
)

# Extrai e exibe a resposta
assistant_response = response.choices[0].message.content
print(f"Resposta da OpenAI:\n{assistant_response}")

# Se o Kestra SDK estiver instalado, pode enviar a resposta como output
try:
    from kestra import Kestra
    
    Kestra.outputs({
        "assistant_response": assistant_response,
        "model_used": response.model,
        "tokens_used": response.usage.total_tokens
    })
    
except ImportError:
    print("Kestra SDK não encontrado. Pulando envio de outputs.")

