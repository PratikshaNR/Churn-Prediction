from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved scaler
with open(r"C:\Users\Pratiksha\Documents\Churn Prediction\models\churn-prediction-standard-scaler-1.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Load the trained model
with open(r"C:\Users\Pratiksha\Documents\Churn Prediction\models\churn-prediction.pkl", 'rb') as file:
    rfc = pickle.load(file)

# Define class labels
classes = ['No churn', 'High Possibility of Churn']

# Define the feature names as used during model training
feature_names = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay']

def churn_prediction(Age, Gender, Tenure, Usage_Frequency, Support_Calls, Payment_Delay):
    # Create a DataFrame with the input features
    sample = pd.DataFrame([[Age, Gender, Tenure, Usage_Frequency, Support_Calls, Payment_Delay]], columns=feature_names)
    
    # Scale the input features
    sample_scaled = scaler.transform(sample)
    sample_scaled_df = pd.DataFrame(sample_scaled, columns=feature_names)
    
    # Make prediction
    prediction = rfc.predict(sample_scaled_df)
    return classes[prediction[0]]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    Age = float(request.form['Age'])
    Gender = float(request.form['Gender'])
    Tenure = float(request.form['Tenure'])
    Usage_Frequency = float(request.form['Usage_Frequency'])
    Support_Calls = float(request.form['Support_Calls'])
    Payment_Delay = float(request.form['Payment_Delay'])
    
    # Make prediction
    prediction = churn_prediction(Age, Gender, Tenure, Usage_Frequency, Support_Calls, Payment_Delay)
    print("Final Prediction:", prediction)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
