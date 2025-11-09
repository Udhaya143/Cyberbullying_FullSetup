# Simple script to train a fallback sklearn LogisticRegression model and save model.pkl
import pandas as pd, joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("models/sample_data.csv")
df = df.dropna(subset=["text","label"])
X = df["text"].astype(str)
y = df["label"].astype(int)
pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X,y)
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "model.pkl")
print("Saved model.pkl")