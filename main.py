import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Step 1: Load the PDF and Create Vector Index ---
pdf_path = "support_doc.pdf"  # Ensure this file is in the root project directory

# Check if file exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# --- Step 2: FastAPI Setup ---
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
    return {"message": "PDF RAG Agent is live."}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        tag = body.get("fulfillmentInfo", {}).get("tag", "")
        user_question = body.get("text", "").strip()

        # Only process if correct webhook tag is matched
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
        if not docs:
            answer = "Sorry, I couldn't find any matching content in the document."
        else:
            answer = qa_chain.run(input_documents=docs, question=user_question)

        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [answer]}}]
            }
        }

    except Exception as e:
        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [f"Error: {str(e)}"]}}]
            }
        }
