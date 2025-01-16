import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
french_stopwords = stopwords.words('french')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
    text = ' '.join(text.split())
    text = ' '.join([word for word in text.split() if word not in french_stopwords])
    return text

train_df['processed_text'] = train_df['Text'].apply(preprocess_text)
test_df['processed_text'] = test_df['Text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['processed_text'])
X_test = vectorizer.transform(test_df['processed_text'])

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, train_df['Label'])

predictions = model.predict(X_test)

test_df['Label'] = predictions

test_df[['Text', 'Label']].to_csv('test_predictions.csv', index=False)
