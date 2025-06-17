import os
import json
import logging
import mlflow
from mlflow.tracking import MlflowClient

# === Configure Logging ===
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            model_info = json.load(f)
        logger.debug("Loaded model info from: %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Failed to load model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register and promote a model in MLflow."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug("Model URI: %s", model_uri)

        model_version = mlflow.register_model(model_uri, model_name)
        logger.debug("Model registered: %s version %s", model_name, model_version.version)

        # Promote to 'Staging'
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info("Model %s version %s transitioned to Staging.", model_name, model_version.version)

    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise


def main():
    try:
        # === Environment variables for DagsHub auth ===
        username = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")

        if not username or not token:
            raise EnvironmentError("Missing DAGSHUB_USERNAME or DAGSHUB_TOKEN in environment variables.")

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # === MLflow Tracking URI ===
        tracking_uri = "https://dagshub.com/mepaluttam/youtube-comment-analysis.mlflow/"
        mlflow.set_tracking_uri(tracking_uri)
        logger.debug("MLflow tracking URI set to: %s", tracking_uri)

        # === Load and register model ===
        model_info = load_model_info("experiment_info.json")
        register_model("yt_chrome_plugin_model", model_info)

    except Exception as e:
        logger.error("Model registration failed: %s", e)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
