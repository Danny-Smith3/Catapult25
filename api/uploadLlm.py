import os
import tarfile
import shutil
import boto3
from transformers import pipeline
from dotenv import load_dotenv

def create_and_upload_llm_to_s3():
    load_dotenv()

    # AWS Setup
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION"),
    )

    BUCKET = "catapult25bullit"
    S3_KEY = "models/llm.h5"

    # === Step 1: Create pipeline
    print("🧠 Loading pipeline...")
    pipe = pipeline(
        "text-generation",
        model="microsoft/phi-1_5",
        device=-1,
        framework="pt"
    )
    print("✅ Pipeline created.")

    # === Step 2: Save model temporarily
    tmp_dir = ".tmp_phi_model"
    os.makedirs(tmp_dir, exist_ok=True)
    pipe.model.save_pretrained(tmp_dir)
    print("✅ Model saved.")
    pipe.tokenizer.save_pretrained(tmp_dir)
    print("✅ Tokenizer saved.")

    # === Step 3: Compress to /models/llm.h5
    os.makedirs("models", exist_ok=True)
    archive_path = "models/llm.h5"

    with tarfile.open(archive_path, "w") as tar:  # uncompressed .tar
        tar.add(tmp_dir, arcname=os.path.basename(tmp_dir)) 
    print("✅ Archive created.")

    # === Step 4: Upload to S3
    print("📤 Uploading to S3...")
    s3.upload_file(archive_path, BUCKET, S3_KEY)
    print("✅ Upload finished.")

    # === Step 5: Cleanup
    shutil.rmtree(tmp_dir)

# Run the function
if __name__ == "__main__":
    create_and_upload_llm_to_s3()
