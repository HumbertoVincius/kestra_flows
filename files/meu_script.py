# Arquivo: files/meu_script.py

import datetime
import sys

# 1. Mensagem simples para o log (stdout)
print("Novo teste de commit no cursor")

# 2. Mostra a data e hora de execução
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Executado em: {current_time}")

# 3. Exemplo de como enviar um valor de volta ao Kestra (usando stdout e um formato específico)
# Kestra é muito flexível, mas o uso da classe 'Kestra' do SDK é recomendado 
# para funcionalidades mais avançadas. Para o básico, o print funciona.

# Se o Kestra SDK estiver instalado (pip install kestra):
try:
    from kestra import Kestra
    
    # Envia uma variável de saída para o Kestra que pode ser referenciada por outras tarefas
    Kestra.outputs({
        "status_message": "Python script finished successfully.",
        "execution_timestamp": current_time
    })
    
except ImportError:
    # Se o Kestra SDK não estiver instalado no ambiente de execução do Python
    sys.stderr.write("Kestra SDK not found. Skipping output feature.\n")
    print("Script concluído sem o Kestra SDK.")
