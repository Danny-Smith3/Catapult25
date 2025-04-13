import yfinance as yf
from stockinfo import get_stock_data
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentimentAI import generate_sentiment_summary, StockSentimentRequest
from transformers import pipeline

app = FastAPI()

# Load the global sentiment analysis model
generator = None

# Allow frontend to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    global generator
    generator = pipeline(
        "text-generation",
        model="microsoft/phi-1_5",
        device=-1,
        framework="pt"
    )

# Get Predictor Model Instance


@app.get("/stock/{ticker}")
async def get_stock_stats(ticker: str):
    data = get_stock_data(ticker)

    if "error" in data:
        return JSONResponse(content=data, status_code=500)

    return JSONResponse(content=data)

@app.get("/predictor/{ticker}")
async def get_stock_graph(ticker: str):

    return f"Received request for prediction graph: {ticker}"

@app.get("/sentiment/{ticker}")
async def get_stock_sentiment(ticker: str):
    request = StockSentimentRequest(ticker=ticker)
    result = generate_sentiment_summary(request)
    return result

