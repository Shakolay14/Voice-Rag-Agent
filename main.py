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

app = FastAPI()

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-key-here"

# Load the PDF and split into chunks
loader = PyPDFLoader("support_doc.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create FAISS vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Load the QA chain
llm = ChatOpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

@app.post("/ask")
async def ask_webhook(request: Request):
    body = await request.json()

    # Extract the user query
    user_question = body.get("text", "")
    
    # Retrieve matching docs
    docs = vectorstore.similarity_search(user_question, k=2)
    
    # Run the QA chain
    response_text = qa_chain.run(input_documents=docs, question=user_question)

    # Format for Dialogflow CX webhook
    return {
        "fulfillment_response": {
            "messages": [
                {
                    "text": {
                        "text": [response_text]
                    }
                }
            ]
        }
    }

# Run server locally (for Render use auto start script)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
