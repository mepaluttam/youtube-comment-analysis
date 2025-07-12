import os
from dotenv import load_dotenv

# === Load environment variables from .env file ===
load_dotenv()

# ✅ Set credentials globally before importing MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
os.environ["DAGSHUB_TOKEN"] = os.getenv("DAGSHUB_TOKEN", "")

import json
import mlflow
import logging
from dagshub import init
from mlflow.tracking import MlflowClient

# === Initialize Dagshub integration ===
init(
    repo_owner="mepaluttam",
    repo_name="youtube-comment-analysis",
    mlflow=True
)

# === Set MLflow tracking URI ===
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/youtube-comment-analysis.mlflow")

# === Logging Configuration ===
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
        logger.debug('✅ Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('❌ File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('❌ Unexpected error loading model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model in MLflow Model Registry and promote to 'Staging'."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"🔁 Registering model from URI: {model_uri}")

        # Register model
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Transition to Staging
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True
        )

        logger.debug(f"✅ Model '{model_name}' version {model_version.version} transitioned to 'Staging'")
        print(f"✅ Model '{model_name}' version {model_version.version} registered and staged successfully.")
    except Exception as e:
        logger.error('❌ Error during model registration: %s', e)
        print(f"❌ Error during registration: {e}")
        raise


def main():
    try:
        model_info_path = "experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)

    except Exception as e:
        logger.error('❌ Model registration failed: %s', e)
        print(f"❌ Registration failed: {e}")


if __name__ == "__main__":
    main()
