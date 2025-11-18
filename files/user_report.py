import os
import json
import re
import uuid
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from supabase import create_client

# === Configuração ===
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID = "639e810b-9d8c-4f31-9569-ecf61fb43888"
CODEGEN_AGENT_NAME = "codegen_agent"
USER_REPORT_AGENT_NAME = "user_report"

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL e SUPABASE_KEY devem estar definidos no .env")

supabase = create_client(supabase_url, supabase_key)

# Usar service key se disponível para escrita
is_using_service = supabase_service_key is not None and supabase_service_key.strip() != ""
write_client = create_client(supabase_url, supabase_service_key) if is_using_service else supabase


def extract_file_paths_from_error(error_text: str) -> List[str]:
    """
    Extrai caminhos de arquivos mencionados no erro.
    Procura por padrões como: app/..., src/..., @/..., etc.
    """
    # Padrões comuns de caminhos em erros
    patterns = [
        r'(app/[^\s:\)]+\.(tsx?|jsx?))',  # app/path/to/file.tsx
        r'(src/[^\s:\)]+\.(tsx?|jsx?))',   # src/path/to/file.ts
        r'(@/[^\s:\)]+\.(tsx?|jsx?))',     # @/path/to/file.tsx
        r'([a-zA-Z0-9_/]+\.(tsx?|jsx?))',  # Qualquer caminho com extensão
    ]
    
    found_paths = set()
    for pattern in patterns:
        matches = re.findall(pattern, error_text)
        for match in matches:
            if isinstance(match, tuple):
                path = match[0]
            else:
                path = match
            # Filtrar paths que parecem válidos (não são apenas extensões)
            if '/' in path and len(path) > 5:
                # Remover @/ e substituir por app/ ou src/ se necessário
                if path.startswith('@/'):
                    # Tentar ambos os caminhos possíveis
                    found_paths.add(path)
                    found_paths.add(path.replace('@/', 'app/'))
                    found_paths.add(path.replace('@/', 'src/'))
                else:
                    found_paths.add(path)
    
    return sorted(list(found_paths))


def get_latest_codegen_ids() -> dict:
    """
    Busca os IDs do último codegen gerado para o projeto.
    """
    try:
        response = (
            supabase.table("codegen_documents")
            .select("codegen_id, prd_id, scaffold_id, schema_id")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if response.data:
            record = response.data[0]
            return {
                "codegen_id": record.get("codegen_id"),
                "prd_id": record.get("prd_id"),
                "scaffold_id": record.get("scaffold_id"),
                "schema_id": record.get("schema_id"),
            }
    except Exception as exc:
        print(f"⚠️  Erro ao buscar último codegen: {exc}")
    
    return {
        "codegen_id": None,
        "prd_id": None,
        "scaffold_id": None,
        "schema_id": None,
    }


def create_tester_report(error_text: str, file_paths: List[str]) -> dict:
    """
    Cria um relatório de teste simulado baseado no erro de runtime.
    Compatível com o formato esperado pelo codegen.
    """
    report = {
        "status": "error",
        "summary": {
            "total_files": len(file_paths) if file_paths else 1,
            "files_with_errors": len(file_paths) if file_paths else 1,
            "files_ok": 0,
            "error_type": "runtime_error"
        },
        "files": []
    }
    
    # Adicionar arquivos com erro
    for file_path in file_paths:
        report["files"].append({
            "file_path": file_path,
            "status": "error",
            "errors": [
                {
                    "type": "runtime_error",
                    "message": error_text[:500] if len(error_text) > 500 else error_text,
                    "line": None,
                    "column": None
                }
            ]
        })
    
    # Se não encontrou arquivos específicos, criar um arquivo genérico
    if not file_paths:
        report["files"].append({
            "file_path": "unknown",
            "status": "error",
            "errors": [
                {
                    "type": "runtime_error",
                    "message": error_text,
                    "line": None,
                    "column": None
                }
            ]
        })
    
    return report


def create_user_correction_request(
    error_text: str,
    file_paths: Optional[List[str]] = None,
    codegen_id: Optional[str] = None,
    prd_id: Optional[str] = None,
    scaffold_id: Optional[str] = None,
    schema_id: Optional[str] = None
) -> bool:
    """
    Cria uma solicitação de correção manual para o codegen.
    
    Args:
        error_text: 
        
        
        file_paths: Lista opcional de arquivos afetados (se None, tenta extrair do erro)
        codegen_id: ID do codegen a corrigir (se None, busca o último)
        prd_id, scaffold_id, schema_id: IDs opcionais (se None, busca do último codegen)
    """
    print("🔧 Criando solicitação de correção manual...")
    
    # Extrair arquivos do erro se não fornecidos
    if file_paths is None:
        file_paths = extract_file_paths_from_error(error_text)
        if file_paths:
            print(f"📋 Arquivos detectados no erro: {len(file_paths)}")
            for path in file_paths[:5]:  # Mostrar até 5
                print(f"   - {path}")
            if len(file_paths) > 5:
                print(f"   ... e mais {len(file_paths) - 5} arquivo(s)")
        else:
            print("⚠️  Nenhum arquivo específico detectado no erro")
    
    # Buscar IDs se não fornecidos
    if not codegen_id:
        ids = get_latest_codegen_ids()
        codegen_id = ids["codegen_id"]
        prd_id = prd_id or ids["prd_id"]
        scaffold_id = scaffold_id or ids["scaffold_id"]
        schema_id = schema_id or ids["schema_id"]
        
        if not codegen_id:
            print("❌ ERRO: Nenhum codegen encontrado. Execute o codegen primeiro.")
            return False
        
        print(f"📦 Usando último codegen: {codegen_id}")
    
    # Criar relatório de teste simulado
    tester_report = create_tester_report(error_text, file_paths)
    
    # Estrutura do content compatível com tester_documents
    content_jsonb = {
        "metadata": {
            "source": "user_report",
            "error_type": "runtime_error"
        },
        "report": tester_report,
        "eslint_output": "",
        "build_output": error_text[:2000] if len(error_text) > 2000 else error_text,
        "raw_output": None
    }
    
    # Salvar relatório em tester_documents
    # O tester_id será gerado automaticamente pelo banco (UUID)
    try:
        tester_doc = {
            "project_id": PROJECT_ID,
            "codegen_id": codegen_id,
            "prd_id": prd_id,
            "scaffold_id": scaffold_id,
            "schema_id": schema_id,
            "content": content_jsonb
        }
        
        result = write_client.table("tester_documents").insert(tester_doc).execute()
        
        if not result.data:
            print("❌ Erro: Relatório de teste não foi criado")
            return False
        
        # Obter tester_id gerado pelo banco
        tester_record = result.data[0]
        tester_id = tester_record.get("tester_id")
        
        if not tester_id:
            print("❌ Erro: tester_id não foi retornado pelo banco")
            return False
        
        print(f"✅ Relatório de teste criado (tester_id: {tester_id})")
    except Exception as exc:
        print(f"❌ Erro ao criar relatório de teste: {exc}")
        return False
    
    # Criar mensagem em agent_messages
    message_content = json.dumps({
        "type": "correction_request",
        "files_with_errors": file_paths if file_paths else [],
        "error_message": error_text[:500],  # Limitar tamanho
        "source": "user_report"
    }, ensure_ascii=False)
    
    try:
        msg_doc = {
            "project_id": PROJECT_ID,
            "from_agent": USER_REPORT_AGENT_NAME,
            "to_agent": CODEGEN_AGENT_NAME,
            "status": "pending",
            "message_content": message_content,
            "prd_id": prd_id,
            "scaffold_id": scaffold_id,
            "schema_id": schema_id,
            "codegen_id": codegen_id,
            "tester_id": tester_id,
        }
        
        result = write_client.table("agent_messages").insert(msg_doc).execute()
        
        if result.data:
            print(f"✅ Mensagem de correção criada com sucesso!")
            print(f"   - Message ID: {result.data[0].get('id')}")
            print(f"   - Codegen ID: {codegen_id}")
            print(f"   - Tester ID: {tester_id}")
            print(f"   - Arquivos afetados: {len(file_paths) if file_paths else 0}")
            print("\n📝 Próximo passo: Execute o codegen para processar a correção.")
            return True
        else:
            print("❌ Erro: Mensagem não foi criada")
            return False
            
    except Exception as exc:
        print(f"❌ Erro ao criar mensagem de correção: {exc}")
        return False


def main():
    """
    Função principal: solicita o erro do usuário e cria a correção.
    """
    print("=" * 60)
    print("🔧 CORREÇÃO MANUAL DE ERROS DE RUNTIME")
    print("=" * 60)
    print("\nCole o erro de runtime abaixo (pressione Enter duas vezes para finalizar):\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        except EOFError:
            break
    
    error_text = "\n".join(lines).strip()
    
    if not error_text:
        print("❌ Erro: Nenhum texto fornecido")
        return
    
    print("\n" + "=" * 60)
    print("📋 Erro recebido:")
    print("=" * 60)
    print(error_text[:500] + ("..." if len(error_text) > 500 else ""))
    print("=" * 60)
    
    # Perguntar se quer especificar arquivos manualmente
    print("\n💡 Deseja especificar arquivos afetados manualmente? (s/N)")
    specify_files = input().strip().lower() == 's'
    
    file_paths = None
    if specify_files:
        print("\nDigite os caminhos dos arquivos (um por linha, Enter vazio para finalizar):")
        file_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            file_paths.append(path)
    
    # Criar correção
    success = create_user_correction_request(error_text, file_paths)
    
    if success:
        print("\n✅ Correção manual criada com sucesso!")
    else:
        print("\n❌ Falha ao criar correção manual")


if __name__ == "__main__":
    main()

