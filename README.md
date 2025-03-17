# LLM/VLLM Chatbot Platform

A comprehensive platform for running text and vision-based language models locally using Ollama and llama.cpp.

## Overview

This project provides a complete solution for deploying and interacting with multiple language models through a clean web interface. It consists of:

- FastAPI backend for handling text and vision requests
- Ollama integration for text-based language models
- llama.cpp integration for vision-based language models
- Docker containerization for easy deployment

## Architecture

The system consists of three main services:

1. **Ollama Service**: Handles text-based LLM requests
2. **llama.cpp Service**: Processes vision-based (multimodal) requests
3. **FastAPI Backend**: Provides API endpoints and orchestrates between services

## Available Models

### Text Models (via Ollama)

- qwen2.5:0.5b
- qwen2.5:1.5b
- phi3.5:3.8b
- qwen2.5:3b
- phi:2.7b
- tinyllama:1.1b

### Vision Models (via llama.cpp)

- Qwen2-VL-2B-Instruct

## Installation

### Prerequisites

- Docker
- Docker Compose

### Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/Hamza-cpp/llm-vllm_platform.git
    cd llm-vllm_platform
    ```

2. Create Docker network:

    ```bash
    docker network create chatbot_network
    ```

3. (Optional) Edit `models.txt` to specify which Ollama models to download during startup.

4. Build and start the services:

    ```bash
    docker-compose up -d
    ```

## API Endpoints

### Text Generation

```text
POST /api/generate
```

Request body:

```json
{
  "context": "Optional context information",
  "user_question": "Your question to the model",
  "model": "qwen2.5:0.5b"
}
```

### Vision Generation

```text
POST /api/generate-vision
```

- Send as form-data:
  - `user_question`: Text question about the image
  - `image`: Image file (.jpg, .jpeg, or .png)

## Development

### Project Structure

- `api.py`: FastAPI backend for text and vision API endpoints
- `ui.py`: Gradio UI for interacting with the models
- `llama_cpp_api.py`: Implementation for vision model using llama.cpp
- `Dockerfile.*`: Docker configurations for each service
- `docker-compose.yaml`: Docker Compose configuration

### Environment Variables

- `OLLAMA_API_URL`: URL for the Ollama API service
- `LLAMACPP_API_URL`: URL for the llama.cpp vision API service
