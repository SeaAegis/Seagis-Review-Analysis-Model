from flask import Flask, request, jsonify
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
from scipy.special import softmax

# Initialize NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Initialize Flask app
app = Flask(__name__)

# Load the saved RoBERTa model and tokenizer
MODEL_DIR = './models/roberta_sentiment'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Load the saved summarizer
summarizer = pipeline("summarization", model='./models/bart_summarizer')

# Define function for RoBERTa-based sentiment scores
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

# Route to analyze reviews
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        reviews = data.get('reviews', [])
        if not reviews:
            return jsonify({"error": "No reviews provided"}), 400
        
        analyzed_reviews, result_list = analyze_reviews(reviews)
        recommendation = recommend_beach(reviews)
        summary = overall_summary(reviews)
        
        response = {
            'recommendation': recommendation,
            'summary': summary
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to analyze reviews
def analyze_reviews(reviews):
    result_list = []
    res = {}
    for i, review in enumerate(reviews):
        vader_result = sia.polarity_scores(review)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
        roberta_result = polarity_scores_roberta(review)
        combined_result = {**vader_result_rename, **roberta_result}
        res[i] = combined_result
        
        if roberta_result['roberta_pos'] > roberta_result['roberta_neg']:
            sentiment = 'Positive'
        elif roberta_result['roberta_neg'] > roberta_result['roberta_pos']:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        result_list.append((review, sentiment))
    
    results_df = pd.DataFrame(res).T
    results_df['Overall_Sentiment'] = [result[1] for result in result_list]
    return results_df.to_dict(), result_list

# Function to recommend visiting the beach
def recommend_beach(reviews):
    _, analyzed_reviews = analyze_reviews(reviews)
    positive_reviews = sum(1 for r in analyzed_reviews if r[1] == 'Positive')
    negative_reviews = sum(1 for r in analyzed_reviews if r[1] == 'Negative')
    if positive_reviews > negative_reviews:
        return "Recommended to visit this beach."
    return "Not recommended to visit this beach."

# Function to summarize reviews
def overall_summary(reviews):
    combined_reviews = ' '.join(reviews)
    try:
        summary = summarizer(combined_reviews, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return combined_reviews

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
