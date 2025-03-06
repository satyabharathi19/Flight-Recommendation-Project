from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved models
with open('recommended_model.pkl', 'rb') as file:
    recommended_model = pickle.load(file)

with open('sentiment_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)

# Load the dataset
df = pd.read_csv('ProcessedDataset.csv')  # Replace with your dataset path

@app.route('/')
def home():
    return render_template('index.html')  # Landing page with search input

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query', '').lower()
    matches = df[df['AirName'].str.lower().str.contains(query, na=False)]['AirName'].unique().tolist()
    return jsonify(matches[:5])  # Return top 5 matching airline names

@app.route('/recommend', methods=['POST'])
def recommend():
    input_airline = request.form['airline_name']
    filtered_flights = df[df['AirName'] == input_airline]

    if not filtered_flights.empty:
        features = ['OverallScore', 'EntertainmentRating', 'FoodRating', 'ServiceRating', 'WifiRating']
        input_features = filtered_flights[features]

        recommended_pred = recommended_model.predict(input_features)
        sentiment_pred = sentiment_model.predict(input_features)

        filtered_flights['Recommended_pred'] = recommended_pred
        filtered_flights['Sentiment_pred'] = sentiment_pred

        recommended_flights = filtered_flights[
            (filtered_flights['Recommended_pred'] == 1) & 
            (filtered_flights['Sentiment_pred'] == 1)
        ]

        if not recommended_flights.empty:
            return render_template('results.html', tables=[recommended_flights[['AirName', 'OverallScore', 'EntertainmentRating', 'ServiceRating', 'Review']].to_html(classes='data')], titles=recommended_flights.columns.values)
        else:
            return render_template('results.html', message=f"No recommended flights with good reviews found for '{input_airline}'.")
    else:
        return render_template('results.html', message=f"No flights found for '{input_airline}'.")

if __name__ == '__main__':
    app.run(debug=True)
