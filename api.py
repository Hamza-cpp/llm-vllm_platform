from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import aiosqlite  # Async SQLite
import requests
import logging
import aiohttp
import os

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama_service:11434/api/generate")
LLAMACPP_API_URL = os.getenv(
    "LLAMACPP_API_URL", "http://llama_cpp_service:8080/generate-vision"
)
DB_PATH = os.getenv("DB_PATH", "./db/llm_responses.db")
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png"}

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
    """Generate a response using the Llama.cpp Vision API service."""

    logger.info(
        f"Sending vision request to llama.cpp service for question: {user_question}"
    )

    try:
        # Create a new aiohttp session
        async with aiohttp.ClientSession() as session:
            # Prepare the form data with the image and question
            form_data = aiohttp.FormData()
            form_data.add_field("user_question", user_question)

            # Read the image file contents
            image_content = await image.read()
            form_data.add_field(
                "image",
                image_content,
                filename=image.filename,
                content_type=image.content_type,
            )

            # Send the request to the llama.cpp API service
            async with session.post(LLAMACPP_API_URL, data=form_data) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get("response", "")

                    # Save response to database
                    async with aiosqlite.connect(DB_PATH) as db:
                        await db.execute(
                            """
                            INSERT INTO responses (context, question, answer)
                            VALUES (?, ?, ?)
                            """,
                            ("Image input", user_question, answer),
                        )
                        await db.commit()

                    return {"response": answer}
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to get response from llama.cpp service: {error_text}"
                    )
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Failed to generate response: {error_text}",
                    )

    except aiohttp.ClientError as e:
        logger.error(f"Connection error to llama.cpp service: {str(e)}")
        raise HTTPException(status_code=503, detail="llama.cpp service unavailable")


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
