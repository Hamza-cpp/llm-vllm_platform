from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import tempfile
import shutil
import logging
import os
import uvicorn
from typing import Set

# Configuration
SUPPORTED_IMAGE_TYPES: Set[str] = {".jpg", ".jpeg", ".png"}
TEMP_DIR = "/app/temp_images"
LLAMA_CPP_PATH = os.getenv(
    "LLAMA_CPP_PATH", "/home/hamza_ok/llama.cpp/build/bin/llama-qwen2vl-cli"
)
QWEN2VL_MODEL_PATH = os.getenv(
    "QWEN2VL_MODEL_PATH",
    "/home/hamza_ok/models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
)
MM_PROJ_PATH = os.getenv(
    "MM_PROJ_PATH",
    "/home/hamza_ok/models/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama.cpp Vision API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "ok", "message": "Llama.cpp Vision API is running!"}


@app.post("/generate-vision")
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
        f"Processing vision request with question: '{user_question}' and image: {image.filename}"
    )

    try:
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
            timeout=500,
        )
    except subprocess.TimeoutExpired:
        os.remove(image_path)  # Cleanup
        logger.error("Process timed out")
        raise HTTPException(status_code=504, detail="Processing timed out")
    except Exception as e:
        os.remove(image_path)  # Cleanup
        logger.error(f"Process error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {str(e)}"
        )

    os.remove(image_path)

    if result.returncode == 0:
        return {"response": result.stdout.strip()}
    else:
        logger.error(f"Llama.cpp error: {result.stderr}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {result.stderr}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
