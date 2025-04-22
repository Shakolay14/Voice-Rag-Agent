import os
import json
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

# Load PDF document
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Create vector database from documents
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

# Load question-answering chain
llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Setup FastAPI app
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
    return {"message": "Voice RAG Agent is live."}

@app.post("/ask")
async def ask_from_doc(request: Request):
    try:
        body = await request.json()
        print("Received body:", json.dumps(body, indent=2))

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
                    "messages": [{"text": {"text": ["No question provided."]}}]
                }
            }

        # Retrieve relevant documents
        docs = db.similarity_search(user_question, k=2)
        if not docs:
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["Sorry, I couldn't find an answer."]}}]
                }
            }

        print("Top doc:", docs[0].page_content[:300])
        answer = qa_chain.run(input_documents=docs, question=user_question)

        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [answer]}}]
            }
        }

    except Exception as e:
        print("Exception:", str(e))
        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [f"Server error: {str(e)}"]}}]
            }
        }
