import os
import boto3
import tarfile
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# Global pipeline var
LLM_PIPELINE = None

def download_and_load_model():
    global LLM_PIPELINE

    # === S3 Setup ===
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION"),
    )

    bucket = "catapult25bullit"
    s3_key = "models/llm.h5"
    local_path = "models/llm.h5"
    model_dir = ".cached_model"

    os.makedirs("models", exist_ok=True)

    print("📥 Downloading llm.h5 from S3...")
    s3.download_file(bucket, s3_key, local_path)
    print("✅ Download complete.")

    # === Extract ===
    print("📦 Extracting model archive...")
    os.makedirs(model_dir, exist_ok=True)
    with tarfile.open(local_path, "r:*") as tar:
        tar.extractall(path=model_dir)
    print("✅ Extraction complete.")

    # === Load model ===
    print("🧠 Loading pipeline...")
    model_subdir = os.path.join(model_dir, os.listdir(model_dir)[0])  # assumes single folder inside .tar
    LLM_PIPELINE = pipeline("text-generation", model=model_subdir, tokenizer=model_subdir, device=-1, framework="pt")
    print("✅ Model ready.")
