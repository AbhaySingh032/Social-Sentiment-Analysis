# 💬 Social Sentiment Analysis – Twitter Edition

This project focuses on analyzing public sentiment from Twitter data using Natural Language Processing (NLP). It aims to classify tweets as positive, negative, or neutral, enabling insights into public opinion on various topics.

**🔍 Main Notebook: twitter-sentiment-analysis.ipynb**


# 🎯 Objectives

🧹 Clean and preprocess raw tweet data

🔠 Apply NLP techniques (tokenization, stopword removal, stemming)

🧠 Train machine learning models to classify tweet sentiments

📊 Visualize sentiment distribution and model performance


# 📁 Dataset

Source: Twitter (pre-collected CSV)

**Columns:**

tweet_id, text, label

Sentiment Labels:

0 → Negative

1 → Neutral

2 → Positive


# 🛠 Tools & Libraries

Python 3.8+

Pandas, NumPy – Data handling

NLTK – Text cleaning and preprocessing

Scikit-learn – Model training & evaluation

Matplotlib, Seaborn – Visualizations


# 🧪 Models Used

Logistic Regression

Naive Bayes

Support Vector Machine (SVM)

Random Forest

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix


# 🔍 Preprocessing Steps

Lowercasing text

Removing URLs, mentions, hashtags, punctuation

Tokenization

Stopword removal

Lemmatization or stemming

TF-IDF vectorization


# 📊 Visualizations

Sentiment distribution (pie/bar chart)

Word clouds for each sentiment class

Confusion matrix

Model accuracy comparison


# 🚀 How to Run

**Clone the repository**
git clone https://github.com/AbhaySingh032/Social-Sentiment-Analysis.git

cd Social-Sentiment-Analysis

**Install dependencies**

pip install -r requirements.txt

**Run the notebook**

jupyter notebook twitter-sentiment-analysis.ipynb

# ✅ requirements.txt (minimal)

pandas==1.5.3

numpy==1.24.3

scikit-learn==1.2.2

nltk==3.8.1

matplotlib==3.7.1

seaborn==0.12.2


# 📦 Future Enhancements

Integrate live Twitter scraping using tweepy

Use deep learning models like LSTM or BERT

Deploy as a Streamlit web app

Track sentiment over time using time series plots

# 👤 Author

Abhay Pal Singh

📧 rabhay032@gmail.com


# ⭐️ Support

If you found this project helpful, please ⭐ the repo and consider sharing it with others!

