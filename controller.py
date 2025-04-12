from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sentimentAI import generate_sentiment_summary, StockSentimentRequest

app = FastAPI()

# Allow frontend to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bull-it.vercel.app/"],  # Replace with your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Predictor Model Instance


@app.get("/stock/{ticker}")
async def get_stock_stats(ticker: str):

    return f"Received request for stock stats: {ticker}"

@app.get("/predictor/{ticker}")
async def get_stock_graph(ticker: str):

    return f"Received request for prediction graph: {ticker}"

@app.get("/sentiment/{ticker}")
async def get_stock_sentiment(ticker: str):
    request = StockSentimentRequest(ticker=ticker)
    result = generate_sentiment_summary(request)
    return result

