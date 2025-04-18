import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Load PDF
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Embed and store
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

# QA Chain setup using LCEL
prompt = PromptTemplate.from_template("Use the following documents to answer the question:\n\n{context}\n\nQuestion: {question}\n\nAnswer:")
llm = ChatOpenAI(temperature=0)
qa_chain = prompt | llm | StrOutputParser()

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
    return {"message": "Voice RAG Agent is live 🎙️"}

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
                    "messages": [
                        {"text": {"text": ["Webhook tag mismatch."]}}
                    ]
                }
            }

        if not user_question:
            return {
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": ["No question found in request."]}}
                    ]
                }
            }

        docs = db.similarity_search(user_question, k=2)
        context = "\n\n".join(doc.page_content for doc in docs)
        print("Top doc:", docs[0].page_content[:300] if docs else "None")

        response_text = qa_chain.invoke({"context": context, "question": user_question})

        return {
            "fulfillment_response": {
                "messages": [
                    {"text": {"text": [response_text]}}
                ]
            }
        }

    except Exception as e:
        print("Error:", e)
        return {
            "fulfillment_response": {
                "messages": [
                    {"text": {"text": [f"Server error: {str(e)}"]}}
                ]
            }
        }
