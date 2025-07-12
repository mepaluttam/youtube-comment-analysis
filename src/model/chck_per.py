import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(f"https://dagshub.com/{username}/youtube-comment-analysis.mlflow")

client = MlflowClient()

try:
    models = client.search_registered_models()
    print("✅ Token has read access to registered models.")
except Exception as e:
    print("❌ Read access failed:", e)

try:
    client.create_registered_model("yt_sentiment_model_test")
    print("✅ Model registration successful.")
except Exception as e:
    print("❌ Model registration failed:", e)
