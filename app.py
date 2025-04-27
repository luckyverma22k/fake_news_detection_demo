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
    # Get the input paragraph
    news_text = request.form['news']  # This gets the 'news' input from HTML form

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

    # Convert to tensor
    input_tensor = torch.tensor(padded, dtype=torch.long).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to label
    prediction = (output > 0.5).float().item()  # threshold at 0.5
    label = 'Fake News' if prediction < 0.5 else 'Real News'

    # Return prediction
    return render_template('index.html', prediction_text=f'Prediction: {label}')


if __name__ == "__main__":
    app.run(debug=True)
