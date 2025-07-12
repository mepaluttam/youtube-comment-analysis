import json
import mlflow
import logging
import os
from dotenv import load_dotenv
from dagshub import init

# === Load environment variables ===
load_dotenv()

# === Required credentials ===
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not (mlflow_username and mlflow_password):
    raise ValueError("❌ MLFLOW_TRACKING_USERNAME and/or MLFLOW_TRACKING_PASSWORD not set in .env")

if not dagshub_token:
    raise ValueError("❌ DAGSHUB_TOKEN not set in .env")

# === Initialize Dagshub (enables MLflow tracking) ===
init(
    repo_owner='mepaluttam',
    repo_name='youtube-comment-analysis',
    mlflow=True
)

# === Set MLflow tracking URI ===
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/youtube-comment-analysis.mlflow")

# === Configure Logging ===
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
    """Load model info from a JSON file."""
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
    """Register the model and transition it to Staging."""
    try:
        # ✅ Explicitly set environment variables (required for model registry access)
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)

        # Promote to staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"✅ Model '{model_name}' version {model_version.version} registered and moved to 'Staging'.")
        print(f"✅ Model '{model_name}' version {model_version.version} registered successfully.")
    except Exception as e:
        logger.error('❌ Error during model registration: %s', e)
        print(f"❌ Error: {e}")
        raise


def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('❌ Model registration failed: %s', e)
        print(f"❌ Registration failed: {e}")


if __name__ == '__main__':
    main()
