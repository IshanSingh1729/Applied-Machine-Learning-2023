from flask import Flask, request, render_template, url_for, redirect
import joblib
import score

app = Flask(__name__)

# Load the trained model
mlp_model = joblib.load("C:\Jupyter Lab\CMI\Applied Machine Learning\Assignment 3\model\multilayer-perceptron-model\model.pkl")

# Set threshold for spam detection
threshold = 0.5

# Define home page route and function
@app.route('/')
def home():
    # Render the spam.html template
    return render_template('spam.html')

# Define spam detection page route and function
@app.route('/spam', methods=['POST'])
def spam():
    # Get sentence from the form submission
    sent = request.form['sent']
    
    # Use the machine learning model to score the sentence
    label, prop = score.score(sent, mlp_model, threshold)
    
    # Determine whether the sentence is spam or not
    lbl = "Spam" if label == 1 else "not a spam"
    
    # Format the result message
    ans = f"""The sentence "{sent}" is {lbl} with propensity {prop} """
    
    # Render the result.html template with the result message
    return render_template('result.html', ans=ans)

# Run the app in debug mode if this file is executed directly
if __name__ == '__main__':
    app.run(debug=True)
