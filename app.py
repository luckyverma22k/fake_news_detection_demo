from flask import Flask, request, jsonify, render_template
import torch
from model import LSTMClassifier   # <-- importing your model class

app = Flask(__name__)

# Load model
model = LSTMClassifier(vocab_size=5000)  # Use same vocab_size you trained with
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Here, get news paragraph from form
    paragraph = list(request.form.values())[0]   # Assuming only 1 input
    # Now you must tokenize, pad it exactly like training time
    # I will show you this part separately if you want

    # Example (pseudo): processed_input = preprocess(paragraph)

    # For now dummy input
    processed_input = torch.randint(0, 5000, (1, 50))  # Shape (batch_size=1, seq_len=50)

    with torch.no_grad():
        output = model(processed_input)
    
    prediction = 'Fake News' if output.item() > 0.5 else 'Real News'

    return render_template('index.html', prediction_text=f'Prediction: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
