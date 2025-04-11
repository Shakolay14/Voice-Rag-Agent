from fastapi import FastAPI, Request
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.qa_with_sources import load_qa_chain
from langchain.llms import OpenAI
import os

app = FastAPI()

# Load and process the PDF once
pdf_path = "support_doc.pdf"  # Adjust if needed
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(pages)

# Create vector store
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embedding)

# LLM setup
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()

    # Handle Dialogflow format
    if "queryResult" in data:
        user_question = data["queryResult"]["queryText"]
    else:
        user_question = data.get("question")

    # Search for relevant chunks
    docs = db.similarity_search(user_question)

    # Get answer from LLM
    answer = chain.run(input_documents=docs, question=user_question)

    # Return to Dialogflow if needed
    if "queryResult" in data:
        return {"fulfillmentText": answer}
    else:
        return {"answer": answer}
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.environ.get("OPENAI_API_KEY")
logger.info(f"API key available: {bool(api_key)}")
