# api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import time
from typing import List, Optional, Dict, Any, Literal
import numpy as np
import google.generativeai as genai
import openai

# Import your existing functions
from app_lang import (
    get_qdrant_client, get_embeddings, get_groq_client,
    qdrant_retriever_tool, stream_answer_from_groq,
    STRICT_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, TOP_K
)

app = FastAPI(title="PMAY-G AI Assistant API")

# CORS middleware to allow requests from n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available models for each provider
AVAILABLE_PROVIDERS = {
    "groq": {
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229"
        ],
        "default_model": "llama-3.1-8b-instant"
    },
    "gemini": {
        "models": [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite"
        ],
        "default_model": "gemini-2.5-flash-lite"
    },
    "openai": {
        "models": [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ],
        "default_model": "gpt-3.5-turbo"
    }
}

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    collection_name: str = "pmay_ocr_docs"
    top_k: int = TOP_K
    provider: Literal["groq", "gemini", "openai"] = "groq"
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 512

class LLMRequest(BaseModel):
    prompt: str
    provider: Literal["groq", "gemini", "openai"] = "groq"
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    max_tokens: int = 512
    system_prompt: str = "You are a helpful and knowledgeable AI assistant. Be clear and concise."

class QueryResponse(BaseModel):
    answer: str
    response_time_ms: float
    retrieved_chunks: List[str] = []
    success: bool
    error: Optional[str] = None
    provider_used: str
    model_used: str

class LLMResponse(BaseModel):
    response: str
    response_time_ms: float
    success: bool
    error: Optional[str] = None
    provider_used: str
    model_used: str

class ProvidersResponse(BaseModel):
    providers: Dict[str, Dict[str, Any]]

# Initialize clients (cached)
@app.on_event("startup")
async def startup_event():
    global qdrant_client, embeddings, groq_client, gemini_client, openai_client
    
    try:
        qdrant_client = get_qdrant_client()
        embeddings = get_embeddings()
        groq_client = get_groq_client()
        
        # Initialize Gemini client
        gemini_api_key = "AIzaSyCz0xghwPk9dZZKsUBcjx08pHNkyDD43K0"
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            gemini_client = genai
        else:
            gemini_client = None
            print("Warning: GOOGLE_API_KEY not set for Gemini")
        
        # Initialize OpenAI client
        openai_api_key = "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"
        if openai_api_key:
            openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            openai_client = None
            print("Warning: OPENAI_API_KEY not set for OpenAI")
            
    except Exception as e:
        print(f"Initialization error: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "PMAY-G AI Assistant API", "status": "healthy"}

@app.get("/providers", response_model=ProvidersResponse)
async def list_available_providers():
    """Return list of available LLM providers and their models"""
    return ProvidersResponse(providers=AVAILABLE_PROVIDERS)

def validate_provider_model(provider: str, model: str) -> Optional[str]:
    """Validate provider and model combination"""
    if provider not in AVAILABLE_PROVIDERS:
        return f"Provider '{provider}' not supported. Available providers: {list(AVAILABLE_PROVIDERS.keys())}"
    
    if model not in AVAILABLE_PROVIDERS[provider]["models"]:
        return f"Model '{model}' not available for provider '{provider}'. Available models: {AVAILABLE_PROVIDERS[provider]['models']}"
    
    return None

def generate_with_groq(prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Generate response using Groq"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    completion = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        stream=False,
    )
    return completion.choices[0].message.content

def generate_with_gemini(prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Generate response using Gemini"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini client not configured")
    
    # Combine system prompt and user prompt
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    model = gemini_client.GenerativeModel(model)
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    return response.text

def generate_with_openai(prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Generate response using OpenAI"""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not configured")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content

def generate_response(provider: str, prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Route to the appropriate provider's generation function"""
    if provider == "groq":
        return generate_with_groq(prompt, system_prompt, model, temperature, max_tokens)
    elif provider == "gemini":
        return generate_with_gemini(prompt, system_prompt, model, temperature, max_tokens)
    elif provider == "openai":
        return generate_with_openai(prompt, system_prompt, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

@app.post("/query", response_model=QueryResponse)
async def query_pmayg(request: QueryRequest):
    start_time = time.perf_counter()
    
    # Validate provider and model
    validation_error = validate_provider_model(request.provider, request.model)
    if validation_error:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return QueryResponse(
            answer="",
            response_time_ms=elapsed_ms,
            retrieved_chunks=[],
            success=False,
            error=validation_error,
            provider_used=request.provider,
            model_used=request.model
        )
    
    try:
        # Simple topic detection for PMAY-related queries
        query_lower = request.question.lower()
        is_pmay_query = any(kw in query_lower for kw in ["pmay", "pmay-g", "pmayg", "pradhan mantri awas yojana"]) \
            or ("housing" in query_lower and "rural" in query_lower)

        retrieved_chunks = []
        retrieved_context = None

        if is_pmay_query:
            # Retrieve from Qdrant
            retrieved_chunks = qdrant_retriever_tool(
                qdrant_client, 
                request.collection_name, 
                embeddings, 
                request.question, 
                k=request.top_k
            )
            
            if retrieved_chunks:
                retrieved_context = "\n\n".join(retrieved_chunks)
            else:
                # If no relevant information
                answer = "Insufficient information available in the provided document to answer your question."
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return QueryResponse(
                    answer=answer,
                    response_time_ms=elapsed_ms,
                    retrieved_chunks=[],
                    success=True,
                    provider_used=request.provider,
                    model_used=request.model
                )
        else:
            # Non-PMAY query
            retrieved_chunks = []

        # Prepare the prompt based on whether we have retrieved context
        if retrieved_context:
            prompt = (
                f"You must answer ONLY using the provided document context.\n"
                f"Question: {request.question}\n\n"
                f"Document context (retrieved from Qdrant_Vector_Store, top-{request.top_k}):\n"
                f"---\n{retrieved_context}\n---\n\n"
                f"Instructions:\n"
                f"- If the context contains the answer, extract it precisely using the same terms and provide a concise answer.\n"
                f"- If the context does not contain the answer, reply with this exact phrase: \"Insufficient information available in the provided document to answer your question.\"\n"
                f"- Do not use outside knowledge. Do not speculate. Keep the answer plain text."
            )
            system_prompt = STRICT_SYSTEM_PROMPT
        else:
            prompt = request.question
            system_prompt = GENERAL_SYSTEM_PROMPT

        # Generate response
        answer = generate_response(
            provider=request.provider,
            prompt=prompt,
            system_prompt=system_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        return QueryResponse(
            answer=answer,
            response_time_ms=elapsed_ms,
            retrieved_chunks=retrieved_chunks,
            success=True,
            provider_used=request.provider,
            model_used=request.model
        )
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return QueryResponse(
            answer="",
            response_time_ms=elapsed_ms,
            retrieved_chunks=[],
            success=False,
            error=str(e),
            provider_used=request.provider,
            model_used=request.model
        )

@app.post("/llm/generate", response_model=LLMResponse)
async def generate_llm_response(request: LLMRequest):
    """General purpose LLM generation endpoint without retrieval"""
    start_time = time.perf_counter()
    
    # Validate provider and model
    validation_error = validate_provider_model(request.provider, request.model)
    if validation_error:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return LLMResponse(
            response="",
            response_time_ms=elapsed_ms,
            success=False,
            error=validation_error,
            provider_used=request.provider,
            model_used=request.model
        )
    
    try:
        # Generate response
        response = generate_response(
            provider=request.provider,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        return LLMResponse(
            response=response,
            response_time_ms=elapsed_ms,
            success=True,
            provider_used=request.provider,
            model_used=request.model
        )
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return LLMResponse(
            response="",
            response_time_ms=elapsed_ms,
            success=False,
            error=str(e),
            provider_used=request.provider,
            model_used=request.model
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)