import os
import boto3
import tarfile
import shutil
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Global list to hold loaded LSTM models
LOADED_MODELS = []

def download_extract_and_load_lstm_models():
    load_dotenv()

    # AWS S3 Setup
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION"),
    )

    bucket = "catapult25bullit"
    # S3 prefix where model archives/files are stored
    prefix = "models/"
    local_models_dir = "models"
    os.makedirs(local_models_dir, exist_ok=True)

    # Temporary cache directory for extraction
    extraction_root = ".cached_models"
    os.makedirs(extraction_root, exist_ok=True)

    # List objects in the S3 bucket under the prefix that end with '.h5'
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        print("No files found under the specified prefix.")
        return

    # Filter out "llm.h5" and only process the .h5 files not named "llm.h5".
    model_files = [
        obj["Key"]
        for obj in response["Contents"]
        if obj["Key"].endswith(".h5") and os.path.basename(obj["Key"]).lower() != "llm.h5"
    ]
    print(f"Found {len(model_files)} .h5 file(s) (excluding llm.h5) to process.")

    for s3_key in model_files:
        filename = os.path.basename(s3_key)
        local_file_path = os.path.join(local_models_dir, filename)

        print(f"\n📥 Downloading {filename} from S3...")
        s3.download_file(bucket, s3_key, local_file_path)
        print(f"✅ Download of {filename} complete.")

        # Check if the file is a tar archive. If so, extract first; otherwise, load directly.
        if tarfile.is_tarfile(local_file_path):
            model_name = os.path.splitext(filename)[0]
            model_extract_dir = os.path.join(extraction_root, model_name)
            os.makedirs(model_extract_dir, exist_ok=True)
            print(f"📦 Extracting {filename} into {model_extract_dir}...")
            try:
                with tarfile.open(local_file_path, "r:*") as tar:
                    tar.extractall(path=model_extract_dir)
                print("✅ Extraction complete.")
                # Look for a .h5 file in the extracted folder.
                extracted_files = [
                    f for f in os.listdir(model_extract_dir)
                    if f.endswith(".h5")
                ]
                if not extracted_files:
                    print(f"⚠️ No .h5 file found inside the archive of {filename}; skipping this model.")
                    continue
                model_file_path = os.path.join(model_extract_dir, extracted_files[0])
            except Exception as e:
                print(f"⚠️ Extraction failed for {filename}: {e}")
                continue
        else:
            # If the file is not a tar archive, assume it's directly the .h5 file.
            model_file_path = local_file_path

        # Load the LSTM model using Keras
        print(f"🧠 Loading LSTM model from {model_file_path}...")
        try:
            lstm_model = load_model(model_file_path)
            LOADED_MODELS.append({
                "name": os.path.splitext(filename)[0],
                "model": lstm_model
            })
            print(f"✅ LSTM model for {filename} loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load LSTM model from {model_file_path}: {e}")
            continue

        # Optionally, clean up the downloaded file if not needed
        try:
            os.remove(local_file_path)
        except Exception as e:
            print(f"⚠️ Could not remove temporary file {local_file_path}: {e}")

    print("\n✅ All available LSTM models processed and loaded. Total loaded models:", len(LOADED_MODELS))


if __name__ == "__main__":
    download_extract_and_load_lstm_models()
