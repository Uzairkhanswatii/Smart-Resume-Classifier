
import streamlit as st
import torch
import joblib
import fitz  # PyMuPDF
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================== Load Model & Tokenizer ==================
model_dir = "/content/drive/MyDrive/Resume_Project/model"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ================== PDF Text Extraction ==================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ================== Prediction ==================
def predict_resume(text):
    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    max_idx = probs.argmax()
    label = label_encoder.inverse_transform([max_idx])[0]
    confidence = probs[max_idx]
    return label, confidence

# ================== Streamlit UI ==================
st.title("Smart Resume Classifier")
uploaded_files = st.file_uploader("Upload one or more PDF resumes", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        label, confidence = predict_resume(text)
        st.write(f"**{uploaded_file.name}** → Predicted Category: {label} (Confidence: {confidence:.2f})")
