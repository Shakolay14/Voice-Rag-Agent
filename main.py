import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Load API key
api_key = os.getenv("OPENAI_API_KEY")

# Load document
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Setup embedding + vectorstore
embedding = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(documents, embedding)

# LLM and QA Chain
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
qa_chain = load_qa_with_sources_chain(llm)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Voice RAG Agent is running."}

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        question = body.get("text", "").strip()
        tag = body.get("fulfillmentInfo", {}).get("tag", "")

        if tag != "ask-doc-question":
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["Invalid webhook tag."]}}]
                }
            }

        if not question:
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["No question provided."]}}]
                }
            }

        docs = db.similarity_search(question, k=2)
        if not docs:
            response_text = "Sorry, I couldn't find an answer in the document."
        else:
            result = qa_chain({"input_documents": docs, "question": question})
            response_text = result["answer"]

        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [response_text]}}]
            }
        }

    except Exception as e:
        return {
            "fulfillment_response": {
                "messages": [{"text": {"text": [f"Error: {str(e)}"]}}]
            }
        }
