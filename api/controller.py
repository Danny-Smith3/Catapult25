from stockinfo import get_stock_data
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentimentAI import generate_sentiment_summary, StockSentimentRequest
from llm_loader import download_and_load_model
from LSTM_helper import compute_sharpe_ratio, compute_ATR, compute_RSI, compute_bollinger_bands, get_predicted_price

app = FastAPI()

# Allow frontend to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    # Load the LLM model at startup
    download_and_load_model()

@app.get("/stock/{ticker}")
async def get_stock_stats(ticker: str):
    data = get_stock_data(ticker)

    if "error" in data:
        return JSONResponse(content=data, status_code=500)

    return JSONResponse(content=data)

@app.get("/predictor/{ticker}")
async def get_stock_graph(ticker: str):
    result = get_predicted_price(ticker)
    return result

@app.get("/sentiment/{ticker}")
async def get_stock_sentiment(ticker: str):
    request = StockSentimentRequest(ticker=ticker)
    result = generate_sentiment_summary(request)
    return result

@app.get("/")
def health_check():
    return {"status": "ok"}

