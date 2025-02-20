from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import logging
import sqlite3  

def init_db():
    conn = sqlite3.connect('llm_responses.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY,
            context TEXT,
            question TEXT,
            answer TEXT,
            rating INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class GenerateRequest(BaseModel):
    context: str
    user_question: str
    model: str = "qwen2.5:1.5b"

class RatingRequest(BaseModel):
    response: str
    rating: int

@app.post("/api/save_rating")
async def save_rating(rating_request: RatingRequest):
    try:
        conn = sqlite3.connect('llm_responses.db')
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE responses
            SET rating = ?
            WHERE answer = ?
        """, (rating_request.rating, rating_request.response))
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_response(request: GenerateRequest):
    full_context = f"Context: {request.context}\n\nQuestion: {request.user_question}"
    logging.info(f"Generating response using model {request.model} with context: {full_context}")

    api_url = "http://localhost:11434/api/generate"
    payload = {
        'model': request.model,
        'prompt': full_context,
        'options': {
        'top_k': 1,
        'top_p': 0.1,
        'temperature': 0.1
        }, 
        "stream" : False
}

    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        generated_response = response.json()
        answer = generated_response.get('response')

        conn = sqlite3.connect('llm_responses.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO responses (context, question, answer)
            VALUES (?, ?, ?)
        """, (request.context, request.user_question, answer))
        conn.commit()
        conn.close()

        return {"response": answer}
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to generate response")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)