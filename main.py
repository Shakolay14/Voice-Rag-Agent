import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the document
pdf_path = "support_doc.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"support_doc.pdf not found at {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Set up embeddings and vector store
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

# Use RetrievalQA chain
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

@app.get("/")
def root():
    return {"message": "Voice RAG Agent is running."}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        print("Received body:", body)

        # Correctly extract tag and user query
        tag = body.get("fulfillmentInfo", {}).get("tag", "")
        user_question = body.get("text", "").strip()

        if tag != "ask-doc-question":
            print("Invalid tag received:", tag)
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["Webhook tag mismatch."]}}]
                }
            }

        if not user_question:
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["No question provided."]}}]
                }
            }

        # Run the RAG chain
        answer = qa_chain.run(user_question)

        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [answer]}}]
            }
        }

    except Exception as e:
        print("Error:", e)
        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [f"Error: {str(e)}"]}}]
            }
        }
