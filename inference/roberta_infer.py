from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("models/roberta_finetuned")
model = AutoModelForSequenceClassification.from_pretrained("models/roberta_finetuned")

def predict_roberta(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    return ["Negative", "Neutral", "Positive"][torch.argmax(outputs.logits).item()]
