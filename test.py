import mlflow
import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# MLflow model loading
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/youtube-comment-analysis.mlflow/")
model_uri = "models:/yt_chrome_plugin_model/6"
model = mlflow.pyfunc.load_model(model_uri)

# Load vectorizer
root_dir = os.getcwd()  # or set your own path
vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')
vectorizer = joblib.load(vectorizer_path)

# Preprocessing function
def preprocess_comment(comment):
    comment = comment.lower().strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
    return comment

# Test samples
test_samples = [
    ("This video is so helpful and inspiring!", 1),
    ("Not sure what I just watched.", 0),
    ("I hate this content, absolute garbage!", -1)
]

comments = [preprocess_comment(text) for text, _ in test_samples]
X_test = vectorizer.transform(comments)
predictions = model.predict(X_test)

# Output predictions
for i, (text, true_label) in enumerate(test_samples):
    print(f"\nOriginal: {text}")
    print(f"Expected Sentiment: {true_label}")
    print(f"Predicted Sentiment: {int(predictions[i])}")

# Classification report
from sklearn.metrics import classification_report
y_true = [label for _, label in test_samples]
y_pred = [int(p) for p in predictions]

print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))
