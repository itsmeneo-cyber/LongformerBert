import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
USE_HF_HUB = os.getenv("USE_HF_HUB", "false").lower() == "true"
MODEL_PATH = os.getenv("HF_MODEL_ID") if USE_HF_HUB else "fine_tuned_longformer"


print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if USE_HF_HUB:
    print("Model and tokenizer loaded successfully from Hugging Face Hub.")
else:
    print("Model and tokenizer loaded successfully from local path.")

# Expose for import
def get_model_and_tokenizer():
    return model, tokenizer
