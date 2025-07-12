import os
import json
import mlflow
import logging
from dagshub import init
from mlflow.tracking import MlflowClient

# === Load .env only when running locally (not in GitHub Actions CI) ===
if os.getenv("GITHUB_ACTIONS") != "true":
    from dotenv import load_dotenv
    load_dotenv()

# === Set required environment variables from either environment or .env ===
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
os.environ["DAGSHUB_TOKEN"] = os.getenv("DAGSHUB_TOKEN", "")

# === Initialize DagsHub (for MLflow + DVC tracking integration) ===
init(
    repo_owner="mepaluttam",
    repo_name="youtube-comment-analysis",
    mlflow=True
)

# === Set MLflow Tracking URI ===
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/youtube-comment-analysis.mlflow")

# === Logger setup ===
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load model info (run_id, model_path) from JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f"✅ Model info loaded from {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error loading model info: {e}")
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model and transition to 'Staging'."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"🔁 Registering model from URI: {model_uri}")

        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True
        )

        logger.info(f"✅ Model '{model_name}' version {model_version.version} registered and staged.")
        print(f"✅ Model '{model_name}' version {model_version.version} registered and staged.")
    except Exception as e:
        logger.error(f"❌ Error during model registration: {e}")
        print(f"❌ Registration error: {e}")
        raise


def main():
    try:
        model_info_path = "experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)

    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        print(f"❌ Registration failed: {e}")


if __name__ == "__main__":
    main()
