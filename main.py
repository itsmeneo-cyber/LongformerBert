from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import logging
import traceback

from loader import get_model_and_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Load model and tokenizer once
logger.info("Loading model and tokenizer...")
model, tokenizer = get_model_and_tokenizer()
device = torch.device("cpu")  # Force CPU
model.to(device)
logger.info("Model and tokenizer loaded successfully.")

# Request model
class CompareAnswersRequest(BaseModel):
    teacher_answer: str
    student_answer: str
    total_marks: float

# Score calculation logic
def calculate_score(entailment: float, neutral: float, contradiction: float) -> float:
    score_ratio = (1.0 * entailment + 0.3 * neutral - 0.1 * contradiction)
    return float(np.clip(score_ratio, 0.0, 1.0))  # Clamp between 0 and 1

# Core sentence comparison function
def compare_sentences(model, tokenizer, premise: str, hypothesis: str) -> float:
    input_text = f"{premise} [SEP] {hypothesis}"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    entailment, neutral, contradiction = probs.tolist()

    logger.info(f"[Comparison Log] Teacher: {premise}")
    logger.info(f"Student: {hypothesis}")
    logger.info(f"Entailment: {entailment:.2f}, Neutral: {neutral:.2f}, Contradiction: {contradiction:.2f}")

    return calculate_score(entailment, neutral, contradiction)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Service is running"}

# Main scoring endpoint
@app.post("/compare_answers")
def compare_answers(request: CompareAnswersRequest):
    try:
        logger.info("Received /compare_answers request")

        if request.total_marks <= 0:
            logger.warning("Invalid total_marks received")
            raise HTTPException(status_code=400, detail="total_marks must be greater than zero.")

        score_ratio = compare_sentences(model, tokenizer, request.teacher_answer, request.student_answer)
        final_score = round(score_ratio * request.total_marks, 2)

        logger.info(f"Final Scaled Score: {final_score} / {request.total_marks}")
        return {"score": float(final_score)}

    except Exception as e:
        logger.error(f"Exception in /compare_answers: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred during comparison.")
