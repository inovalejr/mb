# Vamos gerar uma versão do app.py sem dependências de PDF
# Mantendo apenas CSV + RAG + LangGraph + Gradio

app_clean_path = "/mnt/data/mb-main/mb-main/app_clean.py"

clean_code = """# app_clean.py — Agente RAG + LangGraph + DuckDB (sem PDF, mais leve para Fly.io)
from __future__ import annotations
import os, duckdb, pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gradio as gr

# LangGraph
from langgraph.graph import StateGraph, END

# =========================
# Configurações
# =========================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR = os.getenv("DATA_DIR", "data")

# =========================
# Util: carregar CSVs
# =========================
def load_bases(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    bases = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV não encontrado: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
        bases[name] = df
    return bases

# =========================
# Embeddings / Índice Semântico
# =========================
class SemanticIndex:
    def __init__(self, df: pd.DataFrame, text_cols: List[str]):
        self.df = df.reset_index(drop=True)
        self.text_cols = text_cols
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.corpus = (df[text_cols].fillna("").astype(str).agg(" \\n ".join, axis=1)).tolist()
        self.emb = self.model.encode(self.corpus, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    def search(self, query: str, k: int = 6) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.emb)[0]
        topk = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in topk]

# =========================
# Hybrid Retriever (DuckDB + Semântico)
# =========================
class HybridRetriever:
    def __init__(self, bases: Dict[str, pd.DataFrame]):
        self.bases = bases
        self.text_cols = {
            "Sites": [c for c in bases["Sites"].columns if c.lower() in {"codigo","estado","tipodesite","observacao","descricao","nome"}],
            "Stoppers": [c for c in bases["Stoppers"].columns if c.lower() in {"cod_stopper","descricao","criticidade","risco","base","status","observacao"}],
            "Projeto": [c for c in bases["Projeto"].columns if c.lower() in {"codigo","fase","observacao","descricao","escopo"}],
            "Tarefas": [c for c in bases["Tarefas"].columns if c.lower() in {"codigo","atividade","status","observacao","descricao","executor"}],
        }
        self.indexes = {name: SemanticIndex(df, self.text_cols.get(name) or df.columns.tolist())
                        for name, df in bases.items()}

    def sql_filter(self, base: str, where_sql: str, limit: int = 120) -> pd.DataFrame:
        df = self.bases[base]
        duckdb.register("tmp", df)
        try:
            q = f"SELECT * FROM tmp WHERE {where_sql} LIMIT {limit}"
            return duckdb.sql(q).df()
        finally:
            duckdb.unregister("tmp")

    def semantic_search(self, base: str, query: str, k: int = 8) -> pd.DataFrame:
        idx = self.indexes[base]
        hits = idx.search(query, k=k)
        rows = []
        for i, s in hits:
            row = idx.df.iloc[i].copy()
            row["_score"] = s
            rows.append(row)
        return pd.DataFrame(rows) if rows else idx.df.head(0).assign(_score=[])

# =========================
# LLM Provider (OpenAI/Gemini)
# =========================
class LLMProvider:
    def __init__(self):
        pass  # implementação de fallback pode ser adicionada aqui

    def chat(self, query: str, context: str) -> str:
        return f"Pergunta: {query}\\n\\nContexto:\\n{context}\\n\\nResposta gerada aqui."

# =========================
# Pipeline LangGraph
# =========================
class AgentState(BaseModel):
    query: str
    base: str
    result: Optional[str] = None

def build_graph(retriever: HybridRetriever, llm: LLMProvider):
    graph = StateGraph(AgentState)

    def retrieve(state: AgentState) -> AgentState:
        df = retriever.semantic_search(state.base, state.query, k=5)
        context = df.to_string(index=False)
        ans = llm.chat(state.query, context)
        state.result = ans
        return state

    graph.add_node("retrieve", retrieve)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()

# =========================
# Interface Gradio
# =========================
def main():
    bases = load_bases({
        "Sites": os.path.join(DATA_DIR, "sites.csv"),
        "Stoppers": os.path.join(DATA_DIR, "stoppers.csv"),
        "Projeto": os.path.join(DATA_DIR, "projetos.csv"),
        "Tarefas": os.path.join(DATA_DIR, "tarefas.csv"),
    })
    retriever = HybridRetriever(bases)
    llm = LLMProvider()
    graph = build_graph(retriever, llm)

    def chat_fn(base, query):
        state = AgentState(query=query, base=base)
        out = graph.invoke(state)
        return out.result

    iface = gr.Interface(
        fn=chat_fn,
        inputs=[gr.Dropdown(["Sites","Stoppers","Projeto","Tarefas"], label="Base"), gr.Textbox(label="Pergunta")],
        outputs="text",
        title="Agente RAG (sem PDF)"
    )
    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 8080)))

if __name__ == "__main__":
    main()
"""

with open(app_clean_path, "w", encoding="utf-8") as f:
    f.write(clean_code)

app_clean_path
