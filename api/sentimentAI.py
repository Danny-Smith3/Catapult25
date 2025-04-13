import os
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsdataapi import NewsDataApiClient
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
)

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# NewsData.io setup
newsdata_client = NewsDataApiClient(apikey=NEWSDATA_API_KEY)

class StockSentimentRequest(BaseModel):
    ticker: str

def generate_sentiment_summary(payload: StockSentimentRequest):
    query = f"{payload.ticker} stock"
    response = ""
    try:
        response = newsdata_client.news_api(q=query, language='en', category='business', country='us')
    except Exception as e:
        return {"error": f"News API failed: {str(e)}"}


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
You are a financial analyst AI (don't state you are a AI in the response). Based on the following news headlines and sentiment scores, analyze the market sentiment and provide a short-term outlook for the stock.

Stock: {payload.ticker}

Headlines:
"""
    for a in articles:
        prompt += f'- "{a["title"]}" (Sentiment: {a["sentiment"]}, Source: {a["source_id"]}, Date: {a["pubDate"]})\n'

    prompt += f"\nAverage Sentiment Score: {round(avg_score, 4)}\nConclusion:"

    # Generate summary via LLaMA 3
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7
        )
        raw_output = completion.choices[0].message.content.strip()
        conclusion = raw_output.strip().split("\n\n")[0]
    except Exception as e:
        return {"error": f"Hugging Face API failed: {str(e)}"}

    return {
        "ticker": payload.ticker,
        "average_sentiment": round(avg_score, 4),
        "generated_conclusion": conclusion,
        "articles_analyzed": len(articles),
        "articles": articles
    }