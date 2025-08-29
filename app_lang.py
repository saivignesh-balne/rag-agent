import os
import io
import uuid
import time
from typing import List, Tuple, Generator, Optional
import numpy as np

import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import SearchParams

from groq import Groq


# ==============================
# Configuration & Constants
# ==============================

# Environment variables (set these before running the app)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCz0xghwPk9dZZKsUBcjx08pHNkyDD43K0")
QDRANT_URL = os.getenv("QDRANT_URL", "https://10a0e9fc-b6b2-4326-8f73-a781450e860d.us-west-2-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.U__BJhkOdhV0LrWsfLyOKY7NswPZGed5sSKeSOczGSU")  # optional if Qdrant has no auth

# App constants
APP_TITLE = "PMAY-G AI Assistant"
DEFAULT_COLLECTION_NAME = "mord_test2"
TOP_K = 3
LLM_MODEL = "llama-3.1-8b-instant"  # Groq fast model
STREAM_TEMPERATURE = 0.1  # low temperature for accuracy
MAX_TOKENS = 512  # smaller cap for low latency
CHAT_WINDOW_TURNS = 4  # windowed memory: number of previous turns to include

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 120

# Retrieval tuning
RETRIEVAL_CANDIDATES = 12  # initial candidate pool from Qdrant
MMR_LAMBDA = 0.7  # trade-off between relevance and diversity
SCORE_THRESHOLD = 0.2  # filter very low-similarity hits (cosine similarity)

# Tool definition (strict name to match system prompt)
TOOL_NAME = "Qdrant_Vector_Store"
TOOL_DESCRIPTION = (
    "Use this tool to retrieve information and answer any question about the Pradhan Mantri Awas Yojana – Gramin (PMAY-G) scheme."
)

# System prompts
STRICT_SYSTEM_PROMPT = """You are an AI assistant for the Pradhan Mantri Awas Yojana – Gramin (PMAY-G) scheme. You have access to one tool: 'Qdrant_Vector_Store'.

Rules:
1. To answer any question about PMAY-G, you MUST call the 'Qdrant_Vector_Store' tool with the user's query.
2. Base your final answer strictly and exclusively on the information returned by the tool.
3. If the tool returns no relevant information, you MUST reply with this exact phrase: "Insufficient information available in the provided document to answer your question."
4. Do not use your own knowledge. Do not make up information.
5. Present the final answer in clean, plain text.
"""

GENERAL_SYSTEM_PROMPT = "You are a helpful and knowledgeable AI assistant. Be clear and concise."


# ==============================
# Caching: Clients & Embeddings
# ==============================

@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    # Keep-alive, small timeouts for snappy behavior
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        timeout=5.0,
    )


@st.cache_resource(show_spinner=False)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # Google Generative AI Embeddings: 'models/embedding-001'
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)


@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment.")
    return Groq(api_key=GROQ_API_KEY)


# ==============================
# Utility: PDF -> Text
# ==============================

def extract_text_from_pdf(uploaded_file: "st.UploadedFile") -> str:
    # Efficiently read PDF bytes (no disk I/O)
    data = uploaded_file.read()
    reader = PdfReader(io.BytesIO(data))
    pages_text = []
    for page in reader.pages:
        # PdfReader.extract_text() is fast and sufficient for text-based PDFs
        text = page.extract_text() or ""
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


def chunk_texts(raw_texts: List[str], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    # Balanced chunk size to minimize tokenization + maximize context coverage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: List[str] = []
    for text in raw_texts:
        if text.strip():
            chunks.extend(splitter.split_text(text))
    # Filter out tiny fragments
    return [c.strip() for c in chunks if len(c.strip()) > 30]


# ==============================
# Qdrant: Indexing & Retrieval
# ==============================

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    # Create collection lazily with the first embedding dimension observed
    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(client: QdrantClient, collection_name: str, embeddings: GoogleGenerativeAIEmbeddings, chunks: List[str]) -> Tuple[int, int]:
    # Batch embed for throughput; then upsert to Qdrant
    if not chunks:
        return 0, 0

    # Compute one embedding to determine vector size for collection setup
    probe_vec = embeddings.embed_query("probe")
    ensure_collection(client, collection_name, vector_size=len(probe_vec))

    # Batch embeddings to reduce network overhead; Google embeddings are fast and batched
    batch_size = 64
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = embeddings.embed_documents(batch)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j],
                payload={"text": batch[j]},
            )
            for j in range(len(batch))
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        total += len(points)
    return total, len(chunks)


def qdrant_retriever_tool(client: QdrantClient, collection_name: str, embeddings: GoogleGenerativeAIEmbeddings, query: str, k: int = TOP_K) -> List[str]:
    # Embed the query and fetch an initial candidate pool, then MMR rerank to improve reliability
    query_vec = embeddings.embed_query(query)
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=RETRIEVAL_CANDIDATES,
        with_payload=True,
        with_vectors=True,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )

    # Collect candidates (text, score, vector) and filter very low scores
    candidate_texts: List[str] = []
    candidate_vecs: List[List[float]] = []
    candidate_scores: List[float] = []
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        text = payload.get("text", "")
        score = getattr(hit, "score", 0.0) or 0.0
        vec = getattr(hit, "vector", None)
        if text and vec is not None and score >= SCORE_THRESHOLD:
            candidate_texts.append(text)
            candidate_vecs.append(vec)
            candidate_scores.append(score)

    # Fallback: if filtering removed everything, keep top-k of raw hits by score
    if not candidate_texts:
        results = []
        for hit in hits[:k]:
            payload = getattr(hit, "payload", {}) or {}
            text = payload.get("text", "")
            if text:
                results.append(text)
        return results[:k]

    # MMR reranking to balance relevance and diversity
    selected_indices: List[int] = []
    query_np = np.array(query_vec, dtype=float)
    candidate_np = np.array(candidate_vecs, dtype=float)

    # Normalize for cosine operations
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    query_norm = query_np / (np.linalg.norm(query_np) + 1e-12)
    cand_norm = _normalize(candidate_np)

    similarity_to_query = cand_norm @ query_norm

    # Greedy selection
    selected_set = []
    while len(selected_indices) < min(k, len(candidate_texts)):
        if not selected_indices:
            # pick the most similar to the query first
            idx = int(np.argmax(similarity_to_query))
            selected_indices.append(idx)
            selected_set.append(cand_norm[idx])
            continue

        selected_mat = np.stack(selected_set, axis=0)
        diversity = cand_norm @ selected_mat.T  # (num_candidates, num_selected)
        max_diversity = diversity.max(axis=1)

        mmr_score = MMR_LAMBDA * similarity_to_query - (1 - MMR_LAMBDA) * max_diversity
        # Avoid reselecting already selected
        mmr_score[selected_indices] = -1e9
        idx = int(np.argmax(mmr_score))
        selected_indices.append(idx)
        selected_set.append(cand_norm[idx])

    reranked_texts = [candidate_texts[i] for i in selected_indices]
    return reranked_texts[:k]


# ==============================
# LLM: Streaming Inference
# ==============================

def stream_answer_from_groq(
    groq_client: Groq,
    system_prompt: str,
    user_query: str,
    retrieved_context: Optional[str],
    history: List[Tuple[str, str]],
) -> Generator[str, None, None]:
    """
    Yield assistant tokens as they arrive from Groq.
    Performance notes:
    - Keep messages minimal to reduce prompt processing latency.
    - Temperature 0 for determinism and speed.
    - Max tokens capped to avoid long generations.
    """
    # Construct compact, strict message set.
    # Note: Include only a small window of chat history to minimize latency.
    messages = [{"role": "system", "content": system_prompt}]

    # Include windowed history (if you want follow-up continuity). We instruct the model
    # to base the final answer strictly on the tool output.
    for role, content in history[-(CHAT_WINDOW_TURNS * 2):]:
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    if retrieved_context is not None:
        # Strict tool usage path: embed tool results with the query, force answer only from tool.
        # Improved packaging to steer the model to ground itself in the retrieved text and extract answers precisely.
        user_message = (
            f"You must answer ONLY using the provided document context.\n"
            f"Question: {user_query}\n\n"
            f"Document context (retrieved from {TOOL_NAME}, top-{TOP_K}):\n"
            f"---\n{retrieved_context}\n---\n\n"
            f"Instructions:\n"
            f"- If the context contains the answer, extract it precisely using the same terms and provide a concise answer.\n"
            f"- If the context does not contain the answer, reply with this exact phrase: \"Insufficient information available in the provided document to answer your question.\"\n"
            f"- Do not use outside knowledge. Do not speculate. Keep the answer plain text."
            #f"- Give the answer in hindi."

        )
        messages.append({"role": "user", "content": user_message})
    else:
        # General knowledge path (non-PMAY queries)
        messages.append({"role": "user", "content": user_query})

    # Stream tokens from Groq
    stream = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=STREAM_TEMPERATURE,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ==============================
# Streamlit UI
# ==============================

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role, content)]
    if "indexed" not in st.session_state:
        st.session_state.indexed = False  # whether any docs processed
    if "last_retrieved_chunks" not in st.session_state:
        st.session_state.last_retrieved_chunks: List[str] = [] #type: ignore


def sidebar_ui() -> Tuple[List["st.UploadedFile"], bool, str, bool, int, int]:
    st.sidebar.header("Document Management")
    # Collection controls
    st.sidebar.subheader("Vector Store Collection")
    selected_collection = st.sidebar.text_input(
        label="Collection name",
        value=st.session_state.get("collection_name", DEFAULT_COLLECTION_NAME),
        help="Enter the Qdrant collection to use for retrieval and indexing.",
    )
    create_collection = st.sidebar.checkbox("Create collection if missing", value=True)

    # Chunking controls
    st.sidebar.subheader("Chunking Settings")
    chunk_size = st.sidebar.number_input(
        label="Chunk size (characters)",
        min_value=200,
        max_value=4000,
        value=int(st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        step=100,
        help="Larger chunks can improve answer completeness; smaller chunks improve precision.",
    )
    chunk_overlap = st.sidebar.number_input(
        label="Chunk overlap (characters)",
        min_value=0,
        max_value=1000,
        value=int(st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
        step=20,
        help="Overlap preserves context across adjacent chunks.",
    )

    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs related to PMAY-G. These will be chunked, embedded, and indexed into Qdrant.",
    )
    process = st.sidebar.button("Process Documents", type="primary", use_container_width=True)
    return uploaded_files, process, selected_collection, create_collection, int(chunk_size), int(chunk_overlap)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏠", layout="wide")
    st.title(APP_TITLE)
    st.caption("High-performance agent with strict PMAY-G tool usage for PMAY queries, and general knowledge for other topics.")

    init_session()

    # Clients (cached)
    try:
        qdrant_client = get_qdrant_client()
        embeddings = get_embeddings()
        groq_client = get_groq_client()
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.stop()

    # Sidebar: Upload & Process
    uploaded_files, do_process, selected_collection, create_if_missing, chunk_size, chunk_overlap = sidebar_ui()
    # Keep selected collection in session
    st.session_state.collection_name = selected_collection.strip() or DEFAULT_COLLECTION_NAME
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    if do_process:
        if not uploaded_files:
            st.sidebar.warning("Please upload at least one PDF before processing.")
        else:
            with st.spinner("Processing documents (chunking → embeddings → Qdrant indexing)..."):
                # Extract
                raw_texts = [extract_text_from_pdf(f) for f in uploaded_files]
                # Chunk
                chunks = chunk_texts(raw_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                # Upsert
                inserted, total_chunks = upsert_chunks(
                    client=qdrant_client,
                    collection_name=st.session_state.collection_name,
                    embeddings=embeddings,
                    chunks=chunks,
                )
                st.session_state.indexed = True
            st.sidebar.success(f"Indexed {inserted}/{total_chunks} chunks into collection '{st.session_state.collection_name}'.")

    # Show tool description (for transparency)
    with st.expander("Retrieval Tool", expanded=False):
        st.markdown(f"- Tool name: `{TOOL_NAME}`")
        st.markdown(f"- Description: {TOOL_DESCRIPTION}")
        st.markdown(f"- Vector store: Qdrant collection `{st.session_state.collection_name if 'collection_name' in st.session_state else DEFAULT_COLLECTION_NAME}`, top-k={TOP_K}")

    # Chat history display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about PMAY-G or anything else...")
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Start timing (end-to-end: retrieval + generation)
        start_time = time.perf_counter()

        # Simple topic detection for PMAY-related queries
        query_lower = user_query.lower()
        is_pmay_query = any(kw in query_lower for kw in ["pmay", "pmay-g", "pmayg", "pradhan mantri awas yojana"]) \
            or ("housing" in query_lower and "rural" in query_lower)

        retrieved_chunks: List[str] = []
        retrieved_context: Optional[str] = None

        if is_pmay_query:
            # Strict tool usage for PMAY queries: must retrieve
            retrieved_chunks = qdrant_retriever_tool(qdrant_client, st.session_state.collection_name, embeddings, user_query, k=TOP_K)
            if retrieved_chunks:
                retrieved_context = "\n\n".join(retrieved_chunks)
                st.session_state.last_retrieved_chunks = retrieved_chunks
            else:
                # If no relevant information, respond with the exact insufficiency phrase
                insufficient = "Insufficient information available in the provided document to answer your question."
                with st.chat_message("assistant"):
                    st.write(insufficient)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    st.caption(f"Response time: {elapsed_ms:.0f} ms")
                st.session_state.messages.append({"role": "assistant", "content": insufficient})
                return
        else:
            # Non-PMAY: answer from general knowledge (no retrieval)
            st.session_state.last_retrieved_chunks = []

        # Stream the assistant response token-by-token
        with st.chat_message("assistant"):
            stream = stream_answer_from_groq(
                groq_client=groq_client,
                system_prompt=STRICT_SYSTEM_PROMPT if is_pmay_query else GENERAL_SYSTEM_PROMPT,
                user_query=user_query,
                retrieved_context=retrieved_context if is_pmay_query else None,
                history=[(m["role"], m["content"]) for m in st.session_state.messages],
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            st.caption(f"Response time: {elapsed_ms:.0f} ms")
            final_text = st.write_stream(stream)

            # Compute and show response time
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            st.caption(f"Response time: {elapsed_ms:.0f} ms")

            # Show the top-k retrieved chunks from Qdrant (only when used)
            if retrieved_chunks:
                with st.expander(f"Top-{TOP_K} retrieved chunks from `{st.session_state.collection_name}`", expanded=False):
                    for idx, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"**{idx}.** {chunk}")

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": final_text})


if __name__ == "__main__":
    main()


