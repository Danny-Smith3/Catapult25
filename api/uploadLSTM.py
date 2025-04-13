import os
import tarfile
import shutil
import boto3
from dotenv import load_dotenv

def upload_models_to_s3():
    load_dotenv()

    # AWS Setup
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION"),
    )

    BUCKET = "catapult25bullit"
    LOCAL_MODELS_DIR = r"C:\Users\parth\Documents\Me\Stock_Exploring\models"
    TEMP_ARCHIVE_DIR = ".tmp_archives"

    # Create a temporary directory for archives if it doesn't exist.
    os.makedirs(TEMP_ARCHIVE_DIR, exist_ok=True)

    # Iterate over all items in the models directory
    for item in os.listdir(LOCAL_MODELS_DIR):
        item_path = os.path.join(LOCAL_MODELS_DIR, item)
        if os.path.isdir(item_path):
            # Compress the directory into a tar archive
            archive_name = f"{item}.tar"
            archive_path = os.path.join(TEMP_ARCHIVE_DIR, archive_name)
            print(f"📦 Compressing directory: {item_path}")
            with tarfile.open(archive_path, "w") as tar:
                tar.add(item_path, arcname=item)
            s3_key = f"models/{archive_name}"
            # Upload the archive to S3
            print(f"📤 Uploading archive {archive_name} to s3://{BUCKET}/{s3_key}")
            s3.upload_file(archive_path, BUCKET, s3_key)
            print("✅ Upload finished.")
        elif os.path.isfile(item_path):
            # Directly upload file if it's not a directory
            s3_key = f"models/{item}"
            print(f"📤 Uploading file {item_path} to s3://{BUCKET}/{s3_key}")
            s3.upload_file(item_path, BUCKET, s3_key)
            print("✅ Upload finished.")
        else:
            print(f"⚠️ Skipping unknown item type: {item_path}")

    # Cleanup temporary archives folder after uploads.
    if os.path.exists(TEMP_ARCHIVE_DIR):
        shutil.rmtree(TEMP_ARCHIVE_DIR)
    print("✅ All models uploaded and temporary files cleaned up.")

# Run the function
if __name__ == "__main__":
    upload_models_to_s3()
