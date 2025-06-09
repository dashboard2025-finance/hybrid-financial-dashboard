from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import requests
import pandas as pd
import torch
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FinBERT sentiment pipeline
finbert_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="yiyanghkust/finbert-tone",
    tokenizer="yiyanghkust/finbert-tone"
)
# Load your custom fake news detection model
fake_news_model_path = "D:/RESEARCH PAPERS/SavedModel/fake_news_bert"
tokenizer = BertTokenizer.from_pretrained(fake_news_model_path)
fake_news_model = BertForSequenceClassification.from_pretrained(fake_news_model_path).to(device)
# Function to fetch financial news headlines
def fetch_news_from_api(api_key, category="business", language="en", page_size=20):
    url = f'https://newsapi.org/v2/top-headlines?category={category}&language={language}&apiKey={api_key}&pageSize={page_size}'
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json()['articles']
        headlines = [article['title'] for article in articles if article['title']]
        return headlines
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []
# Fake News Classification
def classify_fake_news(headlines):
    encoding = tokenizer.batch_encode_plus(
        headlines,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = fake_news_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    return ["Fake" if p == 0 else "Real" for p in preds]
from transformers import pipeline
# Sentiment Analysis
def analyze_sentiment(headlines):
    return finbert_sentiment_pipeline(headlines)

# Hybrid Model Execution
def run_hybrid_model(api_key):
    headlines = fetch_news_from_api(api_key)

    if not headlines:
        print("No headlines to analyze.")
        return None

    # Step 1: Sentiment analysis
    sentiments = analyze_sentiment(headlines)

    # Step 2: Fake news classification
    fake_news_predictions = classify_fake_news(headlines)

    # Step 3: Combine results
    df = pd.DataFrame({
        'Headline': headlines,
        'Sentiment': [s['label'] for s in sentiments],
        'Sentiment Score': [s['score'] for s in sentiments],
        'Fake News Prediction': fake_news_predictions
    })

    # Step 4: Optional Hybrid Insight Logic
    df['Hybrid Tag'] = df.apply(lambda row: hybrid_logic(row['Sentiment'], row['Fake News Prediction']), axis=1)

    output_path = "hybrid_model_output.csv"  # or your preferred path
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

    # Show results
    print(df)
    print("\nSentiment Counts:\n", df['Sentiment'].value_counts())
    print("\nFake News Counts:\n", df['Fake News Prediction'].value_counts())
    print("\nHybrid Tags:\n", df['Hybrid Tag'].value_counts())

    return df
# Optional: Rule-based hybrid interpretation
def hybrid_logic(sentiment, prediction):
    if prediction == "Fake" and sentiment == "Positive":
        return "Positive Fake"
    elif prediction == "Fake" and sentiment == "Negative":
        return "Negative Fake"
    elif prediction == "True" and sentiment == "Positive":
        return "Positive True"
    elif prediction == "True" and sentiment == "Negative":
        return "True Negative"
    else:
        return "Neutral"
# Run hybrid model
api_key = 'cfc73062ab434ccb9bb8606e1c8a531a'  # Replace with your valid API key
run_hybrid_model(api_key)
