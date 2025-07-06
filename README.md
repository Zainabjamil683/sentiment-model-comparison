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

## 🗂️ File Structure

sentiment-model-comparison/
├── app.py # Flask backend for inference
├── sentiment_analysis.ipynb # Notebook for training and evaluation
├── sentiment_model_gru.h5 # Trained GRU model (best performing)
├── tokenizer.pkl # Tokenizer saved for inference
├── requirements.txt # All dependencies
├── templates/
│ └── index.html # HTML frontend
├── static/
│ └── style.css # Optional custom styles
└── README.md

