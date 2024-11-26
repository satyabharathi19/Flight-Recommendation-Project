from flask import Flask, request, render_template
import pandas as pd
import pickle

# to WSFC like to cnnect between server and web application
app = Flask(__name__)

# Load the saved models
with open('recommended_model.pkl', 'rb') as file:
    recommended_model = pickle.load(file)

with open('sentiment_model.pkl', 'rb') as file:
    sentiment_model = pickle.load(file)

# Load the dataset
df = pd.read_csv('ProcessedDataset.csv')  # Replace with your dataset path

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # This is the HTML form for user input

# Define the recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input
    input_airline = request.form['airline_name']
    
    # Filter the dataset based on user input
    filtered_flights = df[df['AirName'] == input_airline]

    if not filtered_flights.empty:
        # Extract features for predictio
        features = ['OverallScore', 'EntertainmentRating', 'FoodRating', 'ServiceRating', 'WifiRating']
        input_features = filtered_flights[features]

        # Make predictions using the loaded models
        recommended_pred = recommended_model.predict(input_features)
        sentiment_pred = sentiment_model.predict(input_features)

        # Add predictions to the DataFrame
        filtered_flights['Recommended_pred'] = recommended_pred
        filtered_flights['Sentiment_pred'] = sentiment_pred

        # Filter based on predicted values
        recommended_flights = filtered_flights[
            (filtered_flights['Recommended_pred'] == 1) & 
            (filtered_flights['Sentiment_pred'] == 1)
        ]

        # Render the results on a new HTML page
        if not recommended_flights.empty:
            return render_template('results.html', tables=[recommended_flights[['AirName', 'OverallScore', 'EntertainmentRating', 'ServiceRating', 'Review']].to_html(classes='data')], titles=recommended_flights.columns.values)
        else:
            return f"No recommended flights with good reviews found for '{input_airline}'."
    else:
        return f"No flights found for '{input_airline}'."

if __name__ == '__main__':
    app.run(debug=True)
