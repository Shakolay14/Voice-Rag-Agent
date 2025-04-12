import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

app = FastAPI()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check API key presence
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY")

# Load and prepare the document
pdf_path = "app/support_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create FAISS index
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)

# Load LLM and QA chain
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

@app.get("/")
def root():
    return {"message": "Server is running"}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        tag = body.get("fulfillmentInfo", {}).get("tag")

        # Ensure it's the expected webhook tag
        if tag != "ask-doc-question":
            return JSONResponse(content={
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": ["Unknown webhook tag"]}}
                    ]
                }
            })

        # Extract query from Dialogflow CX format
        user_question = body.get("sessionInfo", {}).get("parameters", {}).get("text")

        if not user_question:
            return JSONResponse(content={
                "fulfillment_response": {
                    "messages": [
                        {"text": {"text": ["No question found in request"]}}
                    ]
                }
            })

        # Search and respond
        matched_docs = db.similarity_search(user_question)
        response = chain.run(input_documents=matched_docs, question=user_question)

        return JSONResponse(content={
            "fulfillment_response": {
                "messages": [
                    {"text": {"text": [response]}}
                ]
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "fulfillment_response": {
                "messages": [
                    {"text": {"text": [f"Error: {str(e)}"]}}
                ]
            }
        })
