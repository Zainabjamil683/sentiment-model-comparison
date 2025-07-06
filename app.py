from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model_gru.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 80  # same as during training

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    confidence = None

    if request.method == 'POST':
        review = request.form['review']
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        pred = model.predict(padded)[0][0]
        sentiment = 'Positive 😊' if pred > 0.5 else 'Negative 😞'
        confidence = round(float(pred) * 100, 2)

    return render_template('index.html', sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
