from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
import traceback

# Initialize FastAPI app
app = FastAPI()

# Load FAISS index and embedding model
embedding = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)

# Load language model and QA chain
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")


@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()

        # Dialogflow CX sends a nested structure — get actual text
        user_question = (
            data.get("text") or
            data.get("queryInput", {}).get("text", {}).get("text") or
            None
        )

        if not isinstance(user_question, str) or not user_question.strip():
            raise ValueError("Invalid or missing 'text' in request.")

        # Search and respond
        docs = db.similarity_search(user_question)
        answer = chain.run(input_documents=docs, question=user_question)

        return {
            "fulfillment_response": {
                "messages": [
                    {
                        "text": {
                            "text": [answer]
                        }
                    }
                ]
            }
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
