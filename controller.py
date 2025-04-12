import yfinance as yf
from stockinfo import get_stock_data
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Get Sentiment Analysis Model Instance

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
async def get_stock_graph(ticker: str):

    return f"Received request for sentimnt analysis: {ticker}"

app.include_router(router)
