from fastapi import FastAPI, HTTPException, Query

from app.embedding_model import EmbeddingModel

app = FastAPI(
    title="spaCy Word Embedding API",
    description="Generate word embeddings using spaCy's en_core_web_lg model.",
    version="1.0.0",
)

embedding_model = EmbeddingModel()


@app.get("/")
def read_root() -> dict:
    """Return basic information about the API."""
    return {
        "message": "Welcome to the spaCy Word Embedding API.",
        "docs": "/docs",
        "embedding_endpoint": "/embedding?word=apple",
    }


@app.get("/health")
def health_check() -> dict:
    """Confirm that the API server and embedding model are available."""
    return {
        "status": "healthy",
        "model": "en_core_web_lg",
    }


@app.get("/embedding")
def get_embedding(
    word: str = Query(..., description="A single word to embed", examples=["apple"]),
) -> dict:
    """Return the embedding vector for one query word."""
    try:
        return embedding_model.get_embedding(word)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error