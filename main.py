from fastapi import FastAPI, Request
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
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
from fastapi import Request, HTTPException

@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        user_question = body.get("text")

        if not isinstance(user_question, str):
            raise HTTPException(status_code=400, detail="Field 'text' must be a string.")

        print("✅ User question:", user_question)

        # Step 1: Search relevant documents
        docs = db.similarity_search(user_question)

        # Step 2: Use RAG chain to answer
        result = qa_chain.invoke({"question": user_question, "context": docs})

        # Step 3: Return Dialogflow-style response
        response = {
            "fulfillment_response": {
                "messages": [
                    {
                        "text": {
                            "text": [result]
                        }
                    }
                ]
            }
        }
        return JSONResponse(content=response)

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

