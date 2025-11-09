import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Example small dataset
texts = [
    "You are stupid",
    "I hate you",
    "You are awesome",
    "Have a great day",
    "You are an idiot",
    "This is a nice comment",
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = bullying, 0 = non-bullying

# Train pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression())
])
model.fit(texts, labels)

# Save the fallback model
joblib.dump(model, "model.pkl")

print("✅ Fallback model trained and saved as model.pkl")
