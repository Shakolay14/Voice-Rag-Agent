import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()

# Load PDF and build vector store
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

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
    return {"message": "Voice RAG Agent is live ✅"}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        print("Received body:", body)

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

        docs = db.similarity_search(user_question, k=2)
        print("Top doc:", docs[0].page_content if docs else "No match found.")

        if not docs:
            answer = "Sorry, I couldn't find an answer in the document."
        else:
            answer = qa_chain.run(input_documents=docs, question=user_question)

        # ✅ Return only plain string as per Dialogflow CX format
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
