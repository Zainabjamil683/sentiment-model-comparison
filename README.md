# 🔍 Deep Learning Comparison for Sentiment Analysis — IMDb Movie Reviews

This project demonstrates a **complete ML pipeline**: data preprocessing, model training, evaluation, and deployment. It compares three deep learning models — **Simple RNN**, **BiLSTM**, and **GRU** — on the IMDb movie reviews dataset for **binary sentiment classification**. The best-performing model (**GRU**) is saved and served via a Flask web interface.

---

## 📌 Project Highlights

- 📊 **Dataset**: 50,000 IMDb movie reviews (positive/negative) from Kaggle
- 🧠 **Models Compared**:
  - Simple RNN
  - Bidirectional LSTM
  - Gated Recurrent Unit (GRU)
- ✅ **Best Model**: GRU (based on validation accuracy)
- 🔐 **Preprocessing**:
  - Lowercasing, HTML & special char removal
  - Stopword removal
  - Stemming with NLTK
  - Tokenization and padding
- 🖥️ **Web Deployment**:
  - Flask web app (`app.py`)
  - Loads the trained GRU model (`sentiment_model_gru.h5`)
  - Loads the saved tokenizer (`tokenizer.pkl`)
  - Simple HTML form interface for input and sentiment prediction

---

## 📁 File Structure

```plaintext
sentiment-model-comparison/
├── app.py                      # Flask backend for inference
├── sentiment_analysis.ipynb    # Notebook for training and evaluation
├── sentiment_model_gru.h5      # Trained GRU model (best performing)
├── tokenizer.pkl               # Tokenizer saved for inference
├── requirements.txt            # All dependencies
├── templates/
│   └── index.html              # HTML frontend
├── static/
│   └── style.css               # Optional custom styles
└── README.md
````

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/zainabjamil683/sentiment-model-comparison.git
cd sentiment-model-comparison
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK corpora (one-time)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🚀 Run the Web App

```bash
python app.py
```

Then open your browser and visit:
🔗 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

Paste any movie review and get the sentiment result instantly!

---

## 📊 Model Summary

| Model      | Accuracy | Notes                        |
| ---------- | -------- | ---------------------------- |
| Simple RNN | \~83%    | Lightweight, fast to train   |
| BiLSTM     | \~85%    | Better context learning      |
| **GRU** ✅  | \~87%    | Best trade-off (used in app) |

---

## 🧠 How It Works

### 🏋️ Training (`sentiment_analysis.ipynb`)

* Loads IMDb dataset using `kagglehub`
* Applies preprocessing: cleaning, stemming, tokenizing
* Trains 3 RNN-based models: RNN, BiLSTM, GRU
* Saves best model (`sentiment_model_gru.h5`) and tokenizer (`tokenizer.pkl`)

### 🌐 Inference (`app.py`)

* Loads GRU model + tokenizer
* Accepts user input via HTML form
* Predicts sentiment with TensorFlow
* Displays result on the web page

---

## ✍️ Author

**Zainab Jamil**
ML/DL Engineer | NLP Enthusiast
📫 [LinkedIn](https://www.linkedin.com/in/zainab-jamil/) *(update with your real link)*

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

* [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [NLTK](https://www.nltk.org/)
* [Flask](https://flask.palletsprojects.com/)

---

## 🚧 Future Improvements

* Add REST API support
* Add live preview using JavaScript
* Deploy using Render/Heroku/Docker
* Try transformer models (e.g., BERT)

```

---
