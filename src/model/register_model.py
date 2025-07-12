import json
import mlflow
import logging
import os
from dotenv import load_dotenv
from dagshub import init  # ✅ Correct import

# === Load environment variables ===
load_dotenv()

# === Initialize Dagshub with MLflow integration ===
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise ValueError("❌ DAGSHUB_TOKEN not found in environment variables or .env file")

init(  # ✅ Use `init` directly
    repo_owner='mepaluttam',
    repo_name='youtube-comment-analysis',
    mlflow=True
)

# === Set MLflow tracking URI ===
mlflow.set_tracking_uri("https://dagshub.com/mepaluttam/youtube-comment-analysis.mlflow")

# === Logging configuration ===
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
    """Register the model to MLflow Model Registry and transition it to Staging."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition to Staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(f"✅ Model '{model_name}' version {model_version.version} registered and transitioned to 'Staging'.")
        print(f"✅ Model '{model_name}' version {model_version.version} registered successfully.")
    except Exception as e:
        logger.error('❌ Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"  # ✅ Fixed typo here
        register_model(model_name, model_info)

    except Exception as e:
        logger.error('❌ Model registration failed: %s', e)
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
