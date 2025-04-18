import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()

# Load PDF
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Embeddings and Vector DB
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

# QA chain using refine (you can also try map_reduce)
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="refine")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Voice RAG Agent is running."}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        print("Received body:", body)

        tag = body.get("fulfillmentInfo", {}).get("tag", "")
        user_question = body.get("text", "").strip()

        if tag != "ask-doc-question":
            print("Invalid tag received:", tag)
            return {
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": ["Webhook tag mismatch."]}}
                    ]
                }
            }

        if not user_question:
            return {
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": ["No question provided."]}}
                    ]
                }
            }

        docs = db.similarity_search(user_question, k=2)

        if not docs:
            response_text = "Sorry, I couldn't find any relevant content in the document."
        else:
            response_text = chain.invoke({"input_documents": docs, "question": user_question})

        return {
            "fulfillment_response": {
                "messages": [
                    {"text": {"text": [response_text]}}
                ]
            }
        }

    except Exception as e:
        print("Webhook error:", e)
        return {
            "fulfillment_response": {
                "messages": [
                    {"text": {"text": [f"Server error: {str(e)}"]}}
                ]
            }
        }
