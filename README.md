# Sentiment Analysis Using NLP

## Project Overview

This project implements a **Sentiment Analysis system using Natural Language Processing (NLP)** on the **Amazon Fine Food Reviews dataset** obtained from Kaggle. The dataset contains approximately **550,000 customer reviews**. The goal of the project is to perform **trinary sentiment classification** by categorizing reviews into **Negative, Neutral, and Positive** sentiments.

Multiple sentiment analysis approaches were implemented, evaluated, and compared, including traditional machine learning, lexicon-based methods, and transformer-based deep learning models. The trained models were then deployed using a **Streamlit web application** for real-time inference.

---

## Objectives

* Perform sentiment analysis on large-scale textual data
* Compare different NLP approaches for sentiment classification
* Evaluate models using standard performance metrics
* Deploy trained models in an interactive web application

---

## Dataset

* **Name:** Amazon Fine Food Reviews Dataset
* **Source:** Kaggle
* **Size:** ~550,000 reviews
* **Content:** User reviews and ratings for food products
* **Labels:** Converted into sentiment classes:

  * Negative
  * Neutral
  * Positive

Note: The dataset folder is excluded from this repository due to size constraints.

---

## Models Implemented

### 1. Logistic Regression (TF-IDF)

* Traditional machine learning approach
* TF-IDF used for feature extraction
* Efficient and scalable for large datasets

### 2. VADER (NLTK)

* Rule-based, lexicon-driven sentiment analyzer
* No training required
* Used to highlight limitations of lexicon-based methods

### 3. Pretrained RoBERTa

* Transformer-based language model
* Uses general language understanding

### 4. Fine-tuned RoBERTa

* RoBERTa model fine-tuned on the Amazon reviews dataset
* Learns domain-specific sentiment patterns

---

## Evaluation Metrics

The models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-score

All evaluation results are stored in a centralized **`metrics.json`** file for easy comparison.

---

## 🗂 Project Structure

```
Sentiment_Analysis_Using_NLP/
│
├── notebooks/
│   ├── logistic_regression.ipynb
│   ├── vader+pre-trained_roberta.ipynb
│   ├── Fine_tune_roberta.ipynb
│   └── Comparison.ipynb
│
├── models/
│   ├── logistic_regression/
│   │   ├── model.pkl
│   │   └── vectorizer.pkl
│   ├── roberta_finetuned/
│   └── VADER/
│
├── inference/
│   ├── logistic_infer.py
│   ├── roberta_infer.py
│   └── vader_infer.py
│
├── metrics.json
├── streamlit_app.py
└── README.md
```

---

## Model Saving & Inference

* All trained models are saved in the `models/` directory
* Inference logic is separated from training code and placed in the `inference/` folder
* This modular design follows industry best practices and simplifies deployment

---

##  Deployment

* A **Streamlit web application** is used for deployment
* Users can enter a review text and select a model
* The selected model predicts and displays the sentiment in real time

---

## Limitations

* Class imbalance in sentiment labels
* VADER is not suitable for trinary sentiment classification
* Limited hyperparameter tuning for transformer models

---

##  Future Scope

* Apply advanced hyperparameter tuning techniques
* Explore more transformer-based models
* Implement aspect-based sentiment analysis
* Extend support to multilingual sentiment analysis
* Deploy the application on cloud platforms

---

## Technologies & Frameworks Used

* **Python**
* **Scikit-learn** (Logistic Regression, TF-IDF)
* **Hugging Face Transformers** (RoBERTa)
* **PyTorch**
* **NLTK (VADER)**
* **Streamlit**

---

## Conclusion

This project demonstrates a complete NLP pipeline, from data preprocessing and model training to evaluation and deployment. By comparing multiple sentiment analysis techniques on a large real-world dataset, 
the project highlights the strengths and limitations of different NLP approaches while delivering a practical, deployable solution.

---

## 👤 Author

*Project developed as part of academic evaluation / capstone project.*
