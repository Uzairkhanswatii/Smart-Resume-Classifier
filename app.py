import streamlit as st
import torch
import fitz  # PyMuPDF
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import io
import json
import re

# ================== Load Model & Tokenizer ==================
st.set_page_config(page_title="Smart Resume Classifier", page_icon="📄")

MODEL_NAME = "uzairkhanswatii/Smart-Resume-Classifier"  # HF Hub model repo
LABEL_ENCODER_URL = "https://raw.githubusercontent.com/Uzairkhanswatii/Smart-Resume-Classifier/main/label_encoder.json"

def load_modelv2():
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Load label encoder from GitHub (JSON instead of pickle)
    response = requests.get(LABEL_ENCODER_URL)
    response.raise_for_status()
    classes = json.loads(response.content.decode("utf-8"))

    # Custom "encoder" wrapper to mimic LabelEncoder API
    class SimpleLabelEncoder:
        def __init__(self, classes):
            self.classes_ = classes
        def inverse_transform(self, indices):
            return [self.classes_[i] for i in indices]

    label_encoder = SimpleLabelEncoder(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, label_encoder, device


# Call once and cache
tokenizer, model, label_encoder, device = load_modelv2()

# ================== Text Preprocessing ==================
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

# ================== PDF Text Extraction ==================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ================== Prediction ==================
def predict_resume(text):
    text = preprocess_text(text)

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

# ================== Recommendation Engine ==================
def recommend_job_roles(predicted_label):
    recommendations = {
        "Data Science": ["Data Analyst", "ML Engineer", "AI Researcher"],
        "Web Developer": ["Frontend Developer", "Backend Developer", "Fullstack Developer"],
        "Software Engineer": ["DevOps Engineer", "System Engineer", "SRE"],
        "Project Manager": ["Scrum Master", "Agile Coach", "Product Manager"],
    }
    return recommendations.get(predicted_label, ["No recommendations available"])

# ================== Streamlit UI ==================
st.title("📄 Smart Resume Classifier")
st.caption("Upload a resume PDF or paste resume text to classify.")

# Input options
option = st.radio("Choose input method:", ["Upload PDF", "Enter Text"])

texts_to_classify = []

if option == "Upload PDF":
    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            texts_to_classify.append((uploaded_file.name, text))

elif option == "Enter Text":
    user_text = st.text_area("Paste your resume text here:")
    if user_text.strip():
        texts_to_classify.append(("Manual Input", user_text))

# Run classification if we have text
if texts_to_classify:
    st.subheader("📌 Classified Resumes")
    with st.container():
        for name, text in texts_to_classify:
            label, confidence = predict_resume(text)
            st.markdown(f"**{name}** → 🏷️ {label} (Confidence: {confidence:.2f})")

            # Show recommendations
            recs = recommend_job_roles(label)
            st.write("🔮 Recommended Roles:", ", ".join(recs))
            st.divider()
