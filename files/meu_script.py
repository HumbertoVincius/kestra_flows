# Arquivo: files/meu_script.py

import datetime
import sys

# 1. Mensagem simples para o log (stdout)
print("testando de outro pc")

# 2. Mostra a data e hora de execução
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Executado em: {current_time}")



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
