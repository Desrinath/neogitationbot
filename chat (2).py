from flask import Flask, request, jsonify
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
import threading

app = Flask(__name__)

# Initialize sentiment analyzer and conversational model
sia = SentimentIntensityAnalyzer()
generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")

@app.route('/negotiate', methods=['POST'])
def negotiate():
    data = request.json
    customer_offer = data.get('offer', '0')
    
    # Analyze sentiment
    sentiment_score = sia.polarity_scores(customer_offer)
    sentiment = 'neutral'
    if sentiment_score['compound'] < -0.05:
        sentiment = 'negative'
    elif sentiment_score['compound'] > 0.05:
        sentiment = 'positive'
    
    # AI negotiation logic
    response_text = f"The customer has offered {customer_offer}. Please negotiate."
    if sentiment == 'negative':
        response_text = "I can see you're not satisfied. How about we drop the price to 75?"
    
    # Generate further responses
    gpt_response = generator(response_text, max_length=50, num_return_sequences=1)
    return jsonify({"bot_response": gpt_response[0]['generated_text'], "sentiment": sentiment})

# Function to run the Flask app in a separate thread
def run_flask_app():
    app.run(debug=True, use_reloader=False)

# Main function to handle user input and send it to the Flask app
def user_input_interaction():
    while True:
        user_input = input("Customer: ")  # Get user input from the terminal
        if user_input.lower() == 'exit':
            break
        # Send the user input to the /negotiate endpoint
        response = requests.post('http://127.0.0.1:5000/negotiate', json={"offer": user_input})
        if response.status_code == 200:
            data = response.json()
            print(f"Bot: {data['bot_response']}")
            print(f"Sentiment: {data['sentiment']}")

if __name__ == '__main__':
    # Run Flask app in a separate thread to allow real-time interaction
    threading.Thread(target=run_flask_app).start()

    # Start interaction with user input
    user_input_interaction()
