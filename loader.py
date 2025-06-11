from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODEL_PATH, USE_HF_HUB



model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# Expose for import
def get_model_and_tokenizer():
    return model, tokenizer
