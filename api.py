from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import requests
import logging
import aiohttp
import os

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama_service:11434/api/generate")
LLAMACPP_API_URL = os.getenv(
    "LLAMACPP_API_URL", "http://llama_cpp_service:8080/generate-vision"
)
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png"}

app = FastAPI(title="Text & Vision API", version="1.1")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    context: str
    user_question: str
    model: str = "qwen2.5:0.5b"


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
