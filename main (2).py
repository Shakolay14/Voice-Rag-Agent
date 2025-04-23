
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os
import uvicorn

# Fetch OpenAI API Key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Load PDF and prepare the vector store
loader = PyPDFLoader("support_doc.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(texts, embeddings)

# Question-answering chain
llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

@app.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    user_question = body.get("text", "")

    if not user_question:
        return JSONResponse(content={"error": "Missing 'text' in request"}, status_code=400)

    docs = vectorstore.similarity_search(user_question, k=3)
    response_text = qa_chain.run(input_documents=docs, question=user_question)

    return JSONResponse(content={
        "fulfillment_response": {
            "messages": [
                {
                    "text": {
                        "text": [response_text]
                    }
                }
            ]
        }
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
