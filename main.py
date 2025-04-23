import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import load_qa_chain

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Load and index the PDF
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(documents, embeddings)

# Initialize LLM and QA chain
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Create FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Voice RAG agent is running."}

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        user_question = body.get("text", "").strip()
        tag = body.get("fulfillmentInfo", {}).get("tag", "")

        if tag != "ask-doc-question":
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["Invalid webhook tag."]}}]
                }
            }

        if not user_question:
            return {
                "fulfillment_response": {
                    "messages": [{"text": {"text": ["No question provided."]}}]
                }
            }

        docs = db.similarity_search(user_question, k=2)
        if not docs:
            answer = "Sorry, I couldn't find an answer in the document."
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
