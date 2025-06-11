from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from loader import get_model_and_tokenizer

app = FastAPI()

print("Loading model and tokenizer...")
model, tokenizer = get_model_and_tokenizer()
print("Model and tokenizer loaded successfully.")

class CompareAnswersRequest(BaseModel):
    teacher_answer: str
    student_answer: str
    total_marks: float

def calculate_score(entailment: float, neutral: float, contradiction: float) -> float:
    # Weighted logic: reward entailment, partial for neutral, slight penalty for contradiction
    score_ratio = (1.0 * entailment + 0.3 * neutral - 0.1 * contradiction)
    return float(np.clip(score_ratio, 0.0, 1.0))  # Clamp between 0 and 1

def compare_sentences(model, tokenizer, premise: str, hypothesis: str) -> float:
    input_text = f"{premise} [SEP] {hypothesis}"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=4096)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    entailment, neutral, contradiction = probs.tolist()  # Convert from numpy to plain float

    print("\n[Comparison Log]")
    print(f"Teacher: {premise}")
    print(f"Student: {hypothesis}")
    print(f"Entailment: {entailment:.2f}, Neutral: {neutral:.2f}, Contradiction: {contradiction:.2f}")

    return calculate_score(entailment, neutral, contradiction)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Service is running"}


@app.post("/compare_answers")
def compare_answers(request: CompareAnswersRequest):
    try:
        print("Received /compare_answers request.")
        if request.total_marks <= 0:
            raise HTTPException(status_code=400, detail="total_marks must be greater than zero.")

        score_ratio = compare_sentences(model, tokenizer, request.teacher_answer, request.student_answer)
        final_score = round(score_ratio * request.total_marks, 2)

        print(f"Final Scaled Score: {final_score} / {request.total_marks}")
        return {"score": float(final_score)}  # Ensure Python float (not numpy)

    except Exception as e:
        print(f"Error occurred during /compare_answers: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during comparison.")
