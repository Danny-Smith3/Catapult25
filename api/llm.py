from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def get_generator():
    return pipeline(
        "text-generation",
        model="tiiuae/falcon-rw-1b",
        device=-1,
        framework="pt"
    )