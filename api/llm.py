from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def get_generator():
    return pipeline(
        "text-generation",
        model="microsoft/phi-1_5",
        device=-1,
        framework="pt"
    )