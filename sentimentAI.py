from pydantic import BaseModel
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsdataapi import NewsDataApiClient
from dotenv import load_dotenv
import os

# Load model for RAG-like summarization/analysis
generator = pipeline(
    "text-generation",
    model="microsoft/phi-1_5",
    framework="pt"
)

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# NewsData.io setup
load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
newsdata_client = NewsDataApiClient(apikey=NEWSDATA_API_KEY)
class StockSentimentRequest(BaseModel):
    ticker: str

def generate_sentiment_summary(payload: StockSentimentRequest):
    query = f"{payload.ticker} stock"
    response = newsdata_client.news_api(q=query, language='en', category='business', country='us')

    if not response.get("results"):
        return {"message": "No news found for this ticker."}

    # Analyze sentiment for each article
    articles = []
    for article in response["results"]:
        title = article["title"]
        description = article.get("description", "")
        content = f"{title}. {description}"
        sentiment = analyzer.polarity_scores(content)["compound"]
        articles.append({     
            "title": title,
            "source_id": article.get("source_id", "Unknown Source"),
            "pubDate": article.get("pubDate", "Unknown Date"),
            "sentiment": sentiment
        })

    avg_score = sum(a['sentiment'] for a in articles) / len(articles)

    # Build prompt for Mixtral
    prompt = f"""
You are a financial analyst AI. Based on the following news headlines and sentiment scores, analyze the market sentiment and provide a short-term outlook for the stock.

Stock: {payload.ticker}

Headlines:
"""
    for a in articles:
        prompt += f'- "{a["title"]}" (Sentiment: {a["sentiment"]}, Source: {a["source_id"]}, Date: {a["pubDate"]})\n'

    prompt += f"\nAverage Sentiment Score: {round(avg_score, 4)}\nConclusion:"

    # Generate conclusion
    result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return {
        "ticker": payload.ticker,
        "average_sentiment": round(avg_score, 4),
        "generated_conclusion": result[0]['generated_text'],
        "articles_analyzed": len(articles)
    }