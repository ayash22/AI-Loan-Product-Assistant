import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from src.rag.rag_query import RAGAssistant
from datetime import datetime

app = FastAPI(title="Loan Product Assistant")

# Initialize RAG assistant
rag = RAGAssistant()

# Template directory
templates = Jinja2Templates(directory="templates")


# ---------------------------------------
# Home Route - Load Chat UI
# ---------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serves the chat UI from templates/index.html
    """
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------
# Chat API - RAG Answer Endpoint
# ---------------------------------------
@app.post("/ask")
async def ask(message: str = Form(...)):
    """
    Receives chat input and returns RAG answer.
    """
    try:
        answer = rag.query(message)
        now = datetime.now().strftime("%H:%M")

        return JSONResponse({
            "text": answer,
            "time": now
        })

    except Exception as e:
        return JSONResponse({
            "text": "Sorry, something went wrong while answering your question.",
            "error": str(e),
            "time": datetime.now().strftime("%H:%M")
        })


# ---------------------------------------
# Run App
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

