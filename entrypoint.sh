#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &  # Start Ollama in the background
sleep 8 


echo "Checking models.txt..."
if [ -f "/models/models.txt" ]; then
    echo "Loading models from models.txt..."
    while IFS= read -r model || [[ -n "$model" ]]; do
        model=$(echo "$model" | tr -d '[:space:]')  # Trim spaces
        if [ -n "$model" ]; then
            if ollama list | grep -q "^$model"; then
                echo "Model $model already exists. Skipping pull."
            else
                echo "Pulling model: $model"
                ollama pull "$model" || echo "Failed to pull $model"
            fi
        fi
    done < /models/models.txt
else
    echo "models.txt not found, skipping model download."
fi

echo "Ollama is ready and running..."
wait  # Keep the Ollama server process running
