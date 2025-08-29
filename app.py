# app.py — Agente RAG + LangGraph + DuckDB + PDF (pronto para Fly.io / Colab)
from __future__ import annotations
import os, duckdb, pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
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
        self.corpus = (df[text_cols].fillna("").astype(str).agg(" \n ".join, axis=1)).tolist()
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
            "Sites":     [c for c in bases["Sites"].columns if c.lower() in {"codigo","estado","tipodesite","observacao","descricao","nome"}],
            "Stoppers":  [c for c in bases["Stoppers"].columns if c.lower() in {"cod_stopper","descricao","criticidade","risco","base","status","observacao"}],
            "Projeto":   [c for c in bases["Projeto"].columns if c.lower() in {"codigo","fase","observacao","descricao","escopo"}],
            "Tarefas":   [c for c in bases["Tarefas"].columns if c.lower() in {"codigo","atividade","status","observacao","descricao","executor"}],
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
        self.use_openai = False
        self.use_gemini = False

        try:
            from openai import OpenAI
            if os.getenv("OPENAI_API_KEY"):
                self.oai = OpenAI()
                self.use_openai = True
        except Exception:
            self.use_openai = False

        try:
            import google.generativeai as genai
            if os.getenv("GEMINI_API_KEY"):
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.gem = genai.GenerativeModel(GEMINI_MODEL)
                self.use_gemini = True
        except Exception:
            self.use_gemini = False

        if not (self.use_openai or self.use_gemini):
            raise RuntimeError("Defina OPENAI_API_KEY ou GEMINI_API_KEY como secret/env.")

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 700) -> str:
        if self.use_openai:
            resp = self.oai.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        # Gemini path
        prompt = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        out = self.gem.generate_content(prompt)
        return (out.text or "").strip()

# =========================
# PDF → Texto
# =========================
def extract_pdf_text(file) -> str:
    if file is None:
        return ""
    text = []
    reader = PdfReader(file.name)
    for page in reader.pages:
        t = page.extract_text() or ""
        if t:
            text.append(t)
    return "\n".join(text)

# =========================
# Estado do Agente (LangGraph)
# =========================
class AgentState(BaseModel):
    question: str
    base_selected: Optional[str] = None
    where_sql: Optional[str] = None
    hybrid_results: Dict[str, Any] = {}
    pdf_text: str = ""
    analysis_notes: str = ""
    answer: Optional[str] = None
    citations: List[str] = []
    errors: List[str] = []

# =========================
# Nós do Agente
# =========================
class AgentNodes:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider):
        self.retriever = retriever
        self.llm = llm

    def route(self, state: Agent
