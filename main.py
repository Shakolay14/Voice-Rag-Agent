import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load and process the PDF
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Embed and index documents
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(documents, embeddings)

# Set up the LLM and chain
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
qa_chain = load_qa_with_sources_chain(llm)

# Set up FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"message": "Document QA webhook is live."}

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        user_question = body.get("text", "").strip()
        tag = body.get("fulfillmentInfo", {}).get("tag", "")

        if tag != "ask-doc-question":
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["Webhook tag mismatch."]}}]
                }
            }

        if not user_question:
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["Please provide a question."]}}]
                }
            }

        docs = db.similarity_search(user_question, k=3)
        result = qa_chain.run({"input_documents": docs, "question": user_question})

        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [result]}}]
            }
        }

    except Exception as e:
        print("Error:", e)
        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [f"Error: {str(e)}"]}}]
            }
        }
