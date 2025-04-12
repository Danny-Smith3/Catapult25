from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/catapult25/pname")

# Get Predictor Model Instance

# Get Sentiment Analysis Model Instance

@app.get("/stock/{ticker}")
async def get_stock_stats(ticker: str):

    return f"Received request for stock stats: {ticker}"

@app.get("/predictor/{ticker}")
async def get_stock_graph(ticker: str):

    return f"Received request for prediction graph: {ticker}"

@app.get("/sentiment/{ticker}")
async def get_stock_graph(ticker: str):

    return f"Received request for sentimnt analysis: {ticker}"

app.include_router(router)
