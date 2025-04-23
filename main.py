from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

# Load API Key from Render's environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Load documents from PDF
loader = PyPDFLoader("support_doc.pdf")
documents = loader.load()

# Embedding and vector DB setup
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(documents, embeddings)

# LLM and QA Chain
llm = OpenAI(openai_api_key=api_key, temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Voice-RAG-Agent is live"}

@app.post("/ask")
async def ask_from_doc(request: Request):
    try:
        body = await request.json()
        tag = body.get("fulfillmentInfo", {}).get("tag", "")
        user_question = body.get("text", "").strip()

        if tag != "ask-doc-question":
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["Webhook tag mismatch."]}}]
                }
            }

        if not user_question:
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["No question found in request."]}}]
                }
            }

        docs = db.similarity_search(user_question, k=2)
        print("Top doc:", docs[0].page_content if docs else "No match found")

        response_text = (
            qa_chain.run(input_documents=docs, question=user_question)
            if docs
            else "Sorry, I couldn't find an answer."
        )

        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [response_text]}}]
            }
        }

    except Exception as e:
        print("Error:", e)
        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [f"Server error: {str(e)}"]}}]
            }
        }
