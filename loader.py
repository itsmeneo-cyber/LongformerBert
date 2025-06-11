from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODEL_PATH, USE_HF_HUB


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
