from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import torch

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data (ensure you have run this at least once)
nltk.download('vader_lexicon')

# Initialize NLTK's SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Initialize the RoBERTa model and tokenizer for sentiment analysis
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize summarizer (BART model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define function for RoBERTa-based sentiment scores
def polarity_scores_roberta(text):
    try:
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
    except Exception as e:
        print(f"Error in RoBERTa sentiment analysis: {str(e)}")
        return {'roberta_neg': 0, 'roberta_neu': 0, 'roberta_pos': 0}

# Function to analyze a list of reviews
def analyze_reviews(reviews):
    result_list = []
    res = {}

    for i, review in enumerate(reviews):
        try:
            # VADER sentiment analysis
            vader_result = sia.polarity_scores(review)
            vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}

            # RoBERTa sentiment analysis
            roberta_result = polarity_scores_roberta(review)
            combined_result = {**vader_result_rename, **roberta_result}
            res[i] = combined_result

            # Determine overall sentiment
            if roberta_result['roberta_pos'] > roberta_result['roberta_neg']:
                sentiment = 'Positive'
            elif roberta_result['roberta_neg'] > roberta_result['roberta_pos']:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            result_list.append((review, sentiment))
        except Exception as e:
            print(f'Error analyzing review {i}: {str(e)}')

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(res).T
    results_df['Overall_Sentiment'] = [result[1] for result in result_list]
    
    return results_df.to_dict(), result_list  # Ensure returning serializable dict

# Aggregating results for recommendation
def recommend_beach(reviews):
    _, analyzed_reviews = analyze_reviews(reviews)

    positive_reviews = sum(1 for r in analyzed_reviews if r[1] == 'Positive')
    negative_reviews = sum(1 for r in analyzed_reviews if r[1] == 'Negative')
    neutral_reviews = sum(1 for r in analyzed_reviews if r[1] == 'Neutral')

    # Voting system: Recommend visiting if positive reviews dominate
    if positive_reviews > negative_reviews:
        recommendation = "Based on the reviews analysis it is Recommended to visit this beach."
    else:
        recommendation = "Based on the reviews analysis it is Not recommended to visit this beach."

    return {
        'recommendation': recommendation
    }

# Summarize reviews
def overall_summary(reviews):
    # Combine all reviews into a single text block
    combined_reviews = ' '.join(reviews)

    # Summarize the combined reviews
    try:
        summary = summarizer(combined_reviews, max_length=140, min_length=50, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Summarization failed: {str(e)}")
        return combined_reviews  # Return original text if summarization fails

# Route to analyze reviews and return sentiment analysis and recommendation
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get reviews from the request
        data = request.get_json()
        reviews = data.get('reviews', [])

        if not reviews:
            return jsonify({"error": "No reviews provided"}), 400

        # Analyze reviews
        analyzed_reviews, result_list = analyze_reviews(reviews)

        # Generate recommendation
        recommendation_result = recommend_beach(reviews)

        # Generate overall summary
        summary_report = overall_summary(reviews)

        # Prepare response
        response = {
            'recommendation': recommendation_result,  # Final beach visit recommendation
            'summary': summary_report  # Summarized review text
        }

        return jsonify(response)
    
    except Exception as e:
        print(f"Error in /analyze route: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port='5001')
