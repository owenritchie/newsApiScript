import os
from newsdataapi import NewsDataApiClient 
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
import datetime
import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv()


def is_text_long_enough(text: str, min_words: int = 0) -> bool:
    if pd.isna(text):
        return False
    return len(text.split()) >= min_words


def process_article(article: Dict)-> Dict:

    x = str(article.get('description', '')) if article.get('description') is not None else '' 
    y = str(article.get('title', '')) if article.get('title') is not None else ''
    content = x+' '+y

    if not is_text_long_enough(content):
        return None 
    
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    tokens = tokenizer.encode(content, return_tensors='pt')
    result = model(tokens)
    sentiment_score = result.logits.tolist()[0]

    print(article.get('author', ''))
    print(content)
    
    return {
        'article_id': article.get('article_id', ''),
        'source': article.get('source_name', ''),
        'category':article.get('category',''),
        'author': article.get('creator', ''),
        'title': article.get('title', ''),
        'url': article.get('link', ''),
        'image_url': article.get('image_url', ''),
        'publish_time': article.get('pubDate', ''),
        'content': content,
        'added_time': datetime.datetime.now().isoformat(),
        'sentiment_score':sentiment_score,
    }

def prepare_articles(raw_news_data: Dict)-> List[Dict]:
    if not raw_news_data.get('results'):
        return
    processed_articles = []
    for article in raw_news_data['results']:
        processed_article = process_article(article)
        processed_articles.append(processed_article)
    return processed_articles

def initialize_firebase():
    """Initialize Firebase by constructing credentials from individual environment variables"""
    try:
        # Check if already initialized
        return firestore.client()
    except ValueError:
        try:
            # Construct the credentials JSON from individual environment variables
            firebase_creds = {
                "type": os.environ.get("FIREBASE_TYPE", "service_account"),
                "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
                "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
                "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
                "auth_uri": os.environ.get("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.environ.get("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.environ.get("FIREBASE_AUTH_PROVIDER_CERT_URL", 
                                                             "https://www.googleapis.com/oauth2/v1/certs"),
                "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL")
            }
            
            # Check if required fields are present
            required_fields = ["project_id", "private_key_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if not firebase_creds.get(field)]
            
            if missing_fields:
                print(f"Missing required Firebase credential fields: {', '.join(missing_fields)}")
                return None
                
            # Write the constructed JSON to a temporary file
            with open('firebase-credentials.json', 'w') as f:
                json.dump(firebase_creds, f)
                
            # Initialize Firebase with the credentials file
            cred = credentials.Certificate('firebase-credentials.json')
            firebase_admin.initialize_app(cred)
            print("Firebase initialized successfully")
            return firestore.client()
        except Exception as e:
            print(f"Firebase initialization error: {str(e)}")
            return None

def upload_to_firestore(articles: List[Dict],collection_name: str = 'articles'):
    db = initialize_firebase()
    batch = db.batch()

    for article in articles:
        doc_ref = db.collection(collection_name).document()
        batch.set(doc_ref,article)

    batch.commit()
    print(f'Successfully uploaded {len(articles)} articles to Firestore!')

def main():
    API_KEY = os.environ.get('NEWS_API_KEY')
    newsapi = NewsDataApiClient(apikey=API_KEY)
    news_data = newsapi.latest_api(
        timezone='America/New_york',
        language='en',
        image = True,
        removeduplicate=True,
        prioritydomain='top',
    )
    processed_articles = prepare_articles(news_data)
    if processed_articles:
        upload_to_firestore(processed_articles)
    else:
        print("No articles found to process")

if __name__ =='__main__':
    main() 
