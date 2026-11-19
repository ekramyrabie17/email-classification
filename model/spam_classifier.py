import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters only
    return re.sub(r"\s+", " ", text).strip()


# 1. Load data
df = pd.read_csv("data\enron_spam_data.csv")

# 2. Preprocess
df["text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")
df["text"] = df["text"].apply(clean_text)
df["label"] = (df["Spam/Ham"] == "spam").astype(int)  # spam=1, ham=0
df = df[df["text"] != ""].drop_duplicates(subset="text")

# 3. Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# 4. Train
model = make_pipeline(
    TfidfVectorizer(stop_words="english", max_features=20000),
    LogisticRegression(max_iter=1000),
)
model.fit(train_df["text"], train_df["label"])

# 5. Evaluate
predictions = model.predict(test_df["text"])
print("Accuracy:", accuracy_score(test_df["label"], predictions))
print(classification_report(test_df["label"], predictions, target_names=["ham", "spam"]))
print(confusion_matrix(test_df["label"], predictions))

# 6. Predict on new emails
for email in ["Congratulations! You won a free prize, click here now",
              "Hi John, please find the meeting notes attached"]:
    result = model.predict([clean_text(email)])[0]
    print(email, "->", "spam" if result == 1 else "ham")
