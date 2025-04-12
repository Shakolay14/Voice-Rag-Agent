from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
import os

app = FastAPI()

# Set your OpenAI API Key (you can use environment variable or hardcode here for test)
from langchain_community.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

# Load the document and prepare the vector store
pdf_path = "support_doc.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
        tag = body.get("fulfillmentInfo", {}).get("tag")
        session = body.get("session", "")
        user_question = body.get("text", "")

        if tag != "ask-doc-question":
            return JSONResponse(
                content={
                    "fulfillment_response": {
                        "messages": [
                            {"text": {"text": [f"Unknown tag: {tag}"]}}
                        ]
                    }
                }
            )

        # Search and respond
        docs = db.similarity_search(user_question)
        answer = chain.run(input_documents=docs, question=user_question)

        return {
            "fulfillment_response": {
                "messages": [
                    {"text": {"text": [answer]}}
                ]
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": [f"Internal error: {str(e)}"]}}
                    ]
                }
            },
        )
