from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import subprocess
import aiosqlite  # Async SQLite
import tempfile
import requests
import logging
import shutil
import os


OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama_service:11434/api/generate")
DB_PATH = os.getenv("DB_PATH", "./db/llm_responses.db")
LLAMA_CPP_PATH = os.getenv(
    "LLAMA_CPP_PATH", "/home/hamza_ok/llama.cpp/build/bin/llama-qwen2vl-cli"
)
QWEN2VL_MODEL_PATH = os.getenv(
    "QWEN2VL_MODEL_PATH",
    "/home/hamza_ok/llama.cpp/model/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
)
MM_PROJ_PATH = os.getenv(
    "MM_PROJ_PATH",
    "/home/hamza_ok/llama.cpp/model/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
)
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="Text & Vision API", version="1.1")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_db():
    """Initialize the SQLite database if it doesn't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context TEXT,
                question TEXT,
                answer TEXT,
                rating INTEGER DEFAULT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        await db.commit()


@app.on_event("startup")
async def startup_event():
    """Run database initialization on startup."""
    await init_db()


class GenerateRequest(BaseModel):
    context: str
    user_question: str
    model: str = "qwen2.5:0.5b"


class RatingRequest(BaseModel):
    response_id: int
    rating: int


@app.get("/api/health", tags=["System"])
async def health_check():
    """Check if the API is running."""
    return {"status": "ok", "message": "API is running!"}


@app.post("/api/generate", tags=["Chatbot"])
async def generate_response(request: GenerateRequest):
    """Generate a response using the Ollama model."""
    full_context = f"Context: {request.context}\n\nQuestion: {request.user_question}"
    logger.info(
        f"Generating response using model {request.model} with context: {full_context}"
    )

    payload = {
        "model": request.model,
        "prompt": full_context,
        "options": {"top_k": 1, "top_p": 0.1, "temperature": 0.1},
        "stream": False,
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        answer = data.get("response", "")

        # Save response to database
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO responses (context, question, answer)
                VALUES (?, ?, ?)
            """,
                (request.context, request.user_question, answer),
            )
            await db.commit()

        return {"response": answer}

    logger.error(f"Failed to generate response: {response.text}")
    raise HTTPException(
        status_code=response.status_code, detail="Failed to generate response"
    )


@app.post("/api/generate-vision", tags=["Chatbot"])
async def generate_vision_response(
    user_question: str = Form(...), image: UploadFile = File(...)
):
    """Generate a response using Llama.cpp with Qwen2-VL (Vision Model)."""
    # Get the file extension
    ext = os.path.splitext(image.filename)[-1].lower()

    # Validate image format
    if ext not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. Supported formats: {', '.join(SUPPORTED_IMAGE_TYPES)}",
        )
    # Create a temp file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_image:
        shutil.copyfileobj(image.file, temp_image)
        image_path = temp_image.name

    logger.info(
        f"Processing vision request with question: {user_question} and image: {image.filename}"
    )

    result = subprocess.run(
        [
            LLAMA_CPP_PATH,
            "-m",
            QWEN2VL_MODEL_PATH,
            "--mmproj",
            MM_PROJ_PATH,
            "-p",
            user_question,
            "--image",
            image_path,
        ],
        capture_output=True,
        text=True,
    )

    os.remove(image_path)  # Cleanup

    if result.returncode == 0:
        return {"response": result.stdout.strip()}
    else:
        logger.error(f"Llama.cpp error: {result.stderr}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


@app.post("/api/save_rating", tags=["Chatbot"])
async def save_rating(request: RatingRequest):
    """Save a rating for a response."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                UPDATE responses SET rating = ? WHERE id = ?
            """,
                (request.rating, request.response_id),
            )
            await db.commit()

        return {"status": "success", "message": "Rating saved"}
    except Exception as e:
        logger.error(f"Error saving rating: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving rating")


@app.get("/api/responses", tags=["Database"])
async def get_responses(limit: Optional[int] = 10):
    """Retrieve the last N responses."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT * FROM responses ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()

    return [
        {
            "id": row[0],
            "context": row[1],
            "question": row[2],
            "answer": row[3],
            "rating": row[4],
            "timestamp": row[5],
        }
        for row in rows
    ]


@app.delete("/api/responses/{response_id}", tags=["Database"])
async def delete_response(response_id: int):
    """Delete a response by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM responses WHERE id = ?", (response_id,))
        await db.commit()

    return {"status": "success", "message": f"Response {response_id} deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
