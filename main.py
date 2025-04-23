from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import load_qa_chain
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI()

@app.post("/ask")
async def ask_from_doc(request: Request):
    body = await request.json()
    user_question = body.get("text", "")

    loader = PyPDFLoader("support_doc.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)
    docs = db.similarity_search(user_question)

    llm = ChatOpenAI(api_key=api_key, temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    response_text = qa_chain.invoke({"input_documents": docs, "question": user_question})

    return JSONResponse(content={
        "fulfillment_response": {
            "messages": [
                {"text": {"text": [response_text['output_text']]}}
            ]
        }
    })

@app.get("/")
def health_check():
    return {"status": "ok"}

