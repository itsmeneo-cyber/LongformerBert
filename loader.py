import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

USE_HF_HUB = os.getenv("USE_HF_HUB", "false").lower() == "true"
HF_MODEL_ID = os.getenv("HF_MODEL_ID")
LOCAL_MODEL_PATH = os.getenv("MODEL_PATH", "model")  # default to ./model

if USE_HF_HUB:
    print(f"[Loader] Loading model from Hugging Face Hub: {HF_MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
else:
    print(f"[Loader] Loading model locally from path: {LOCAL_MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

def get_model_and_tokenizer():
    return model, tokenizer
