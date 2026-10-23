
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
print("Loading datasets...")

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Check required columns
for filename, df in [("Fake.csv", fake), ("True.csv", real)]:
    if "text" not in df.columns:
        raise ValueError(f"'text' column missing in {filename}")

# Create labels: 0 = Fake, 1 = Real
fake = fake[["text"]].copy()
fake["label"] = 0

real = real[["text"]].copy()
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Clean empty rows and duplicates
data["text"] = data["text"].fillna("").astype(str).str.strip()
data = data[data["text"].str.len() > 30]
data = data.drop_duplicates(subset=["text"])

print("Total samples:", len(data))
print("Dataset loaded successfully!")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

# Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# Train model
print("\nTraining model. Please wait...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["FAKE", "REAL"]
))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/fake_news_model.joblib",
    compress=3
)

print("\nModel saved successfully!")
print("File: models/fake_news_model.joblib")

# Sample prediction
sample = [
    "The government announced a new policy today, according to an official statement."
]

prediction = model.predict(sample)[0]
scores = model.predict_proba(sample)[0]

print("\nSample prediction:", "REAL NEWS" if prediction == 1 else "FAKE NEWS")
print("Model score:", round(max(scores) * 100, 2), "%")
print("\nTraining completed successfully!")
