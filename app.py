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

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # Download label_encoder.pkl from Hugging Face Hub
    label_path = hf_hub_download(repo_id=MODEL_NAME, filename="label_encoder.pkl")
    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)

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
