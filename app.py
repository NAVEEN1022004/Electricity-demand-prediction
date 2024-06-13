from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = pickle.load(open("model/xgb_Week 15, Apr 2019.sav", "rb"))

# Function to preprocess input data
def preprocess_data(data):
    # Extract the features needed for prediction
    features = [
        'week_X-2', 'week_X-3', 'week_X-4', 'MA_X-4', 'dayOfWeek', 'weekend',
        'holiday', 'Holiday_ID', 'hourOfDay', 'T2M_toc'  # Add all required features
    ]

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(data, index=[0])  # Assuming input data is a dictionary

    # Perform any necessary preprocessing steps
    # For example:
    # 1. Convert datetime strings to datetime objects
    input_df['datetime'] = pd.to_datetime(input_df['datetime'])

    # 2. Extract additional datetime features if needed
    input_df['dayOfWeek'] = input_df['datetime'].dt.dayofweek
    input_df['hourOfDay'] = input_df['datetime'].dt.hour

    # 3. Drop unnecessary columns
    input_df.drop(['datetime'], axis=1, inplace=True)

    # 4. Handle missing values if any
    input_df.fillna(0, inplace=True)  # Replace NaNs with 0, for example

    # 5. Scale or normalize the features if required
    # Fit the scaler to your training data
    scaler = MinMaxScaler()  # Assuming MinMaxScaler was used during training
    
    # Load your training data
    # Replace `your_training_data.xlsx` with the path to your actual training data file
    your_training_data = pd.read_excel("your_training_data.xlsx")
    
    # Extract the features used for training
    X_train = your_training_data[['week_X-2', 'week_X-3', 'week_X-4', 'MA_X-4', 'dayOfWeek', 'weekend',
                                  'holiday', 'Holiday_ID', 'hourOfDay', 'T2M_toc']]  # Use all required features
    
    # Fit the scaler to the training data
    scaler.fit(X_train)

    # Transform the input data using the fitted scaler
    input_df_scaled = scaler.transform(input_df)

    return input_df_scaled

# Initialize Flask application
app = Flask(__name__)

# Define route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Define route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from form
    input_data = {
        'datetime': request.form['datetime'],
        'week_X-2': float(request.form['week_X-2']),
        'week_X-3': float(request.form['week_X-3']),
        'week_X-4': float(request.form['week_X-4']),
        'MA_X-4': float(request.form['MA_X-4']),  # Add all required features
        'dayOfWeek': float(request.form['dayOfWeek']),
        'weekend': float(request.form['weekend']),
        'holiday': float(request.form['holiday']),
        'Holiday_ID': float(request.form['Holiday_ID']),
        'hourOfDay': float(request.form['hourOfDay']),
        'T2M_toc': float(request.form['T2M_toc'])
    }

    # Preprocess input data
    processed_data = preprocess_data(input_data)

    # Make prediction
    prediction = model.predict(processed_data)

    # Return prediction to HTML template
    return render_template("result.html", prediction=prediction)

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
