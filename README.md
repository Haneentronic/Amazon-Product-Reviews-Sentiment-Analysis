# 📦 Amazon Product Reviews – Sentiment Analysis

This project is part of my NLP internship with **Elevvo**, focusing on classifying the sentiment (positive or negative) of customer reviews from Amazon using traditional machine learning techniques.

## 🧠 Objective

To build a text classification model that can determine whether a product review expresses **positive** or **negative** sentiment.

---

## 📊 Dataset

We used a dataset containing Amazon product reviews, with each entry labeled as either **positive** or **negative** sentiment. You can find similar datasets on:

- [Amazon Reviews Dataset (Kaggle)](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- IMDb Reviews (alternative for sentiment datasets)

---

## 🛠️ Tools & Libraries

- Python
- Pandas
- Scikit-learn
- NLTK
- CountVectorizer / TF-IDF
- Naive Bayes (MultinomialNB)

---

## ⚙️ Workflow

1. **Data Preprocessing**
   - Lowercasing
   - Removing stopwords, punctuation
   - Tokenization

2. **Feature Engineering**
   - Convert text to numerical vectors using `CountVectorizer` or `TfidfVectorizer`

3. **Model Training**
   - Train a **Naive Bayes classifier** on the vectorized data

4. **Evaluation**
   - Evaluate with accuracy, confusion matrix, precision, recall, and F1 score

---

## 📈 Results

The Naive Bayes classifier achieved high accuracy and fast performance, making it suitable for baseline models in sentiment analysis tasks.

---

## 🖥️ Sample Code

```python
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.pipeline import make_pipeline
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2)

# Create model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train and evaluate
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(classification_report(y_test, preds))

📚 Learnings

  Practical implementation of traditional NLP pipelines

  Importance of preprocessing and feature engineering

  Building explainable models for sentiment analysis

🚀 Future Improvements

    Use pre-trained transformer models (like BERT)

    Deploy the model as a web service

    Extend classification to multi-class or aspect-based sentiment

📌 Status

✅ Completed – This was the first task in the Elevvo NLP Internship.

🔗 Project Link
🔗 Kaggle ((https://www.kaggle.com/code/hanenebrahim/amazon-product-reviwes-sentiment-analysis/edit/run/248707546)





