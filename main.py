from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai

app = FastAPI()

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    
    # Check if request is from Dialogflow
    if "queryResult" in data:
        user_question = data["queryResult"]["queryText"]
    else:
        user_question = data.get("question")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_question}]
    )
    answer = response.choices[0].message["content"]

    # If Dialogflow, wrap in fulfillment format
    if "queryResult" in data:
        return {
            "fulfillmentText": answer
        }
    else:
        return {"answer": answer}
