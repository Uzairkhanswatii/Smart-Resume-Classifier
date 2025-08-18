import streamlit as st
import torch
import fitz  # PyMuPDF
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
from huggingface_hub import hf_hub_download

# ================== Load Model & Tokenizer ==================
st.set_page_config(page_title="Smart Resume Classifier", page_icon="📄")

MODEL_NAME = "uzairkhanswatii/Smart-Resume-Classifier"  # HF Hub model repo
LABEL_ENCODER_URL = "https://raw.githubusercontent.com/uzairkhanswatii/Smart-Resume-Classifier/main/label_encoder.pkl"

@st.cache_resource
def load_model():
    import requests, io, pickle
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Load label encoder from GitHub
    LABEL_ENCODER_URL = "https://raw.githubusercontent.com/uzairkhanswatii/Smart-Resume-Classifier/main/label_encoder.pkl"
    response = requests.get(LABEL_ENCODER_URL)
    response.raise_for_status()
    label_encoder = pickle.load(io.BytesIO(response.content))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, label_encoder, device

tokenizer, model, label_encoder, device = load_model()

# ================== PDF Text Extraction ==================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
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

# ================== Recommendation Engine (Optional) ==================
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
st.caption("Upload one or more PDF resumes and get predicted categories with confidence scores.")

uploaded_files = st.file_uploader(
    "Choose PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    st.subheader("📌 Classified Resumes")
    with st.container():
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            label, confidence = predict_resume(text)
            st.markdown(f"**{uploaded_file.name}** → 🏷️ {label} (Confidence: {confidence:.2f})")

            # Show recommendations (optional)
            recs = recommend_job_roles(label)
            st.write("🔮 Recommended Roles:", ", ".join(recs))
            st.divider()
