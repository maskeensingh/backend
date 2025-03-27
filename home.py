import os
import json
import logging
import warnings
import streamlit as st
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import google.generativeai as genai
import re

# ------------------- Streamlit & Core Setup -------------------
st.set_page_config(
    page_title="Skin Disease Diagnosis Assistant",
    layout="centered",
    initial_sidebar_state="auto",
)

# ------------------- Logging & Env Setup -------------------
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
load_dotenv()

# ------------------- Load Trained Model -------------------
class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(SkinDiseaseCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

@st.cache_resource
def load_trained_model(model_path="skin_disease_model.pth", num_classes=11):
    """Load the trained model with saved weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please upload it.")
        st.stop()

    model = SkinDiseaseCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

# Load model
model, device = load_trained_model()

# ------------------- Image Preprocessing -------------------
def preprocess_image(image):
    """Preprocess the uploaded image before passing it to the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

# ------------------- Class Labels -------------------
CLASS_NAMES = [
    'Unknown',  
    'Eczema',
    'Warts Molluscum and other Viral Infections',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma (BCC)',
    'Melanocytic Nevi (NV)',
    'Benign Keratosis-like Lesions (BKL)',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections'
]

# ------------------- Image-Based Disease Prediction -------------------
def predict_disease_from_image(image):
    """Predict disease from an uploaded image using the trained model."""
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = predicted_idx.item()
        confidence_score = confidence.item()
    
    disease = CLASS_NAMES[predicted_class] if 0 <= predicted_class < len(CLASS_NAMES) else "Unknown Disease"

    return {"disease": disease, "score": round(confidence_score, 2)}

# ------------------- Google Gemini AI Setup -------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API Key is missing. Check your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_response(prompt):
    """Generate a response using Google Gemini AI."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# ------------------- Text-Based Disease Prediction -------------------
def predict_disease_from_text(description):
    """Use Google Gemini AI to predict diseases based on user input."""
    prompt = f"""
    You are a medical expert. Based on the following description, predict the top 5 possible skin diseases.

    User Description: {description}

    Return ONLY a valid JSON list in the format:
    [
        {{"disease": "Disease Name", "score": 0.8}},
        {{"disease": "Another Disease", "score": 0.6}}
    ]
    """

    response_text = get_gemini_response(prompt)

    match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            diseases = json.loads(json_text)
            if isinstance(diseases, list) and all(isinstance(d, dict) for d in diseases):
                return diseases
        except json.JSONDecodeError:
            pass

    return []

# ------------------- Generate Follow-Up Questions & Store in Session -------------------
if "followup_questions" not in st.session_state:
    st.session_state.followup_questions = []

def generate_followup_questions(predictions):
    """Generate follow-up questions only once and store in session."""
    if not st.session_state.followup_questions:
        prompt = f"""
        Given the following possible skin diseases based on text and image inputs:
        {json.dumps(predictions, indent=2)}

        What are the most relevant medical questions to ask the user to refine the final diagnosis? 
        Provide 3-5 questions in plain text.
        """
        response = get_gemini_response(prompt)
        st.session_state.followup_questions = response.split("\n")

# ------------------- Predict Final Disease -------------------
def get_final_disease_prediction(predictions, user_answers):
    """Send final user responses + predictions to Gemini AI to determine the disease."""
    prompt = f"""
    Given these initial predictions:
    {json.dumps(predictions, indent=2)}

    And the user's answers to follow-up questions:
    {json.dumps(user_answers, indent=2)}

    Based on this information, determine the most likely skin disease. Return ONLY the final disease name in plain text.
    """

    response = get_gemini_response(prompt)
    return response.strip()

# ------------------- Streamlit UI -------------------
st.title("Skin Disease Diagnosis Assistant")

with st.sidebar:
    st.subheader("User Inputs")
    user_description = st.text_area("Describe your skin problem:", value="", help="E.g. 'Red rash with itchiness for 3 days'")
    uploaded_image = st.file_uploader("Upload an image (optional):", type=["jpg", "jpeg", "png"])

final_predictions = []
if user_description.strip():
    disease_predictions = predict_disease_from_text(user_description)
    if disease_predictions:
        final_predictions.extend(disease_predictions)
        st.json(disease_predictions)

if uploaded_image:
    image_prediction = predict_disease_from_image(uploaded_image)
    final_predictions.append(image_prediction)
    st.json(image_prediction)

if final_predictions:
    generate_followup_questions(final_predictions)

user_answers = {}
if st.session_state.followup_questions:
    for i, q in enumerate(st.session_state.followup_questions):
        user_answers[f"Q{i+1}"] = st.text_input(q, key=f"question_{i+1}")

if all(user_answers.values()) and user_answers:
    if st.button("Get Final Diagnosis"):
        final_disease = get_final_disease_prediction(final_predictions, user_answers)
        st.success(f"Final Predicted Disease: **{final_disease}**")

st.info("Powered by **Google Gemini AI** and **Deep Learning Model** for accurate skin disease diagnosis.")
