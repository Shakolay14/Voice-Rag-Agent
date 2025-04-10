
from fastapi import FastAPI
from pydantic import BaseModel
import openai

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query.question}]
    )
    return {"answer": response.choices[0].message["content"]}
