import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

BUCKET = "catapult25bullit"

def upload_all_models():
    folder = "models"
    for filename in os.listdir(folder):
        local_path = os.path.join(folder, filename)
        s3_path = f"models/{filename}"
        s3.upload_file(local_path, BUCKET, s3_path)

if __name__ == "__main__":
    upload_all_models()