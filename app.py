import os
import json
import time
import requests
import google.generativeai as genai
from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
from dotenv import load_dotenv

# ------------------- Setup -------------------
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")  # For session management

socketio = SocketIO(app, cors_allowed_origins="*")

# Load Google Gemini API Key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Google Gemini API Key is missing! Set 'GOOGLE_API_KEY' in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# Google Maps API key for clinics lookup
MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# ------------------- Text Prediction using Gemini -------------------
def predict_disease_from_text(description):
    prompt = f"""
You are a medical expert. Predict the top 5 possible skin diseases based on this description:
'{description}'
Return JSON format:
[
    {{"disease": "Disease Name", "score": 0.8}},
    {{"disease": "Another Disease", "score": 0.6}}
]
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return []

def generate_followup_questions(predictions):
    prompt = f"""
Given these possible skin diseases based on the text input:
{json.dumps(predictions, indent=2)}
Generate 3-5 follow-up medical questions to refine the final diagnosis.
Return ONLY plain text questions separated by new lines.
    """
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text
    questions = response.strip().split("\n")
    return [q for q in questions if q.strip() != ""]

# ------------------------------------------------------------------------------
#  API Endpoints
# ------------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    text_description = request.form.get("description")
    # Any image file provided will be ignored.
    final_predictions = []

    if text_description:
        text_predictions = predict_disease_from_text(text_description)
        final_predictions.extend(text_predictions)

    followup_questions = generate_followup_questions(final_predictions)

    return jsonify({
        "predictions": final_predictions,
        "followup_questions": followup_questions
    })

@app.route("/api/final-diagnosis", methods=["POST"])
def final_diagnosis_api():
    data = request.json
    predictions = data.get("predictions", [])
    user_answers = data.get("user_answers", {})

    prompt = f"""
Based on these AI predictions:
{json.dumps(predictions, indent=2)}
And user responses:
{json.dumps(user_answers, indent=2)}
Determine the final skin disease.
Return ONLY the final disease name in plain text.
    """
    
    final_disease = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text.strip()

    treatment_prompt = f"""
You are a medical assistant. Provide a structured and easy-to-understand treatment plan for the following skin condition:

**Disease:** {final_disease}

Respond using this exact format. Each section should have **2–3 short bullet points**. Keep the explanations **simple, practical, and relevant for a general audience**.

**Diagnosis:** [Short explanation of the disease and how it's usually identified]

**Symptoms:**
• [Common symptom 1]
• [Common symptom 2]
• [Common symptom 3]

**Causes:**
• [Major cause or risk factor]
• [Another common contributing factor]

**Treatments (Ordered):**
• Ayurvedic Solutions: [1–2 natural treatments with brief benefits]
• Home Remedies: [1–2 things people can try at home for relief]
• Non-Prescription Medications: [1–2 OTC products with when to use them]
• Prescription Medications: [1–2 doctor-prescribed options and their purpose]

**When to See a Doctor:**
• [Early warning sign]
• [Progression or worsening symptom]
    """
    
    try:
        treatment_response = genai.GenerativeModel("gemini-2.0-flash").generate_content(treatment_prompt).text.strip()
    except Exception:
        treatment_response = "⚠️ Unable to fetch treatment details. Please consult a dermatologist."
    
    return jsonify({
        "final_disease": final_disease,
        "treatment": treatment_response
    })

@app.route("/api/find_clinics", methods=["POST"])
def find_clinics():
    data = request.json
    user_location = data.get("location")
    range_km = data.get("range", 20)
    radius = range_km * 1000

    if isinstance(user_location, str):
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={user_location}&key={MAPS_API_KEY}"
        geocode_response = requests.get(geocode_url)
        geocode_data = geocode_response.json()
        if geocode_data.get("status") != "OK":
            return jsonify({"error": "Unable to geocode location"}), 400
        location = geocode_data["results"][0]["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]
    else:
        lat, lng = user_location.get("lat"), user_location.get("lng")
    
    categories = {
        "NGO": "NGO hospital",
        "Government": " skin government hospital",
        "Private": "skin private hospital"
    }
    
    clinics = []
    for category, keyword in categories.items():
        places_url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"location={lat},{lng}&radius={radius}&type=hospital&keyword={keyword}&key={MAPS_API_KEY}"
        )
        places_response = requests.get(places_url)
        places_data = places_response.json()
        
        for place in places_data.get("results", []):
            place_id = place.get("place_id")
            details_url = (
                f"https://maps.googleapis.com/maps/api/place/details/json?"
                f"place_id={place_id}&fields=name,formatted_address,formatted_phone_number,opening_hours,website,rating&key={MAPS_API_KEY}"
            )
            details_response = requests.get(details_url)
            details_data = details_response.json().get("result", {})
            
            clinic = {
                "category": category,
                "name": place.get("name"),
                "place_id": place.get("place_id"),
                "address": details_data.get("formatted_address"),
                "phone": details_data.get("formatted_phone_number"),
                "website": details_data.get("website"),
                "rating": place.get("rating"),
                "location": place.get("geometry", {}).get("location", {}),
                "hours": details_data.get("opening_hours", {}).get("weekday_text", [])
            }
            clinics.append(clinic)
    
    sorted_order = ["NGO", "Government", "Private"]
    clinics.sort(key=lambda x: sorted_order.index(x["category"]) if x["category"] in sorted_order else 999)
    
    return jsonify({"clinics": clinics})

@app.route("/api/health_chat", methods=["POST"])
def health_chat():
    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    prompt = f"""
You are a women's health assistant. A user asked: {question}

Reply with a **short, concise answer (1–2 sentences max)**. 
Focus on:
• Quick explanation
• 1–2 key tips
• When to see a doctor

If the question isn't about women's health, politely decline.
Return your response in **plain text** only.
    """
    
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text.strip()
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Socket.IO Endpoints -------------------
@socketio.on("join")
def on_join(data):
    room = data.get("room")
    if room:
        join_room(room)
        emit("room_joined", {"message": f"User joined room {room}"}, to=room)
    else:
        emit("error", {"message": "Room name not provided"})

@socketio.on("signal")
def on_signal(data):
    room = data.get("room")
    signalData = data.get("signalData")
    if room and signalData:
        emit("signal", {"signalData": signalData}, to=room, skip_sid=request.sid)
    else:
        emit("error", {"message": "Missing room or signalData"})

# ------------------- Webpage Rendering Endpoints -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_description = request.form.get("description")
        # Any image file provided is ignored.
        final_predictions = []
        
        if text_description:
            text_predictions = predict_disease_from_text(text_description)
            final_predictions.extend(text_predictions)
        
        session["predictions"] = final_predictions
        session["followup_questions"] = generate_followup_questions(final_predictions)
        session["detection_mode"] = "Text Only"
        return redirect(url_for("followup"))
    
    return render_template("index.html")

@app.route("/followup", methods=["GET", "POST"])
def followup():
    if "followup_questions" not in session or not session["followup_questions"]:
        return redirect(url_for("index"))
    
    if request.method == "POST":
        user_answers = {q: request.form.get(q, "No answer provided") for q in session["followup_questions"]}
        session["user_answers"] = user_answers
        return redirect(url_for("final_diagnosis_page"))
    
    return render_template("followup.html", questions=session["followup_questions"])

@app.route("/final_diagnosis", methods=["GET"])
def final_diagnosis_page():
    if "predictions" not in session or "user_answers" not in session:
        return redirect(url_for("index"))
    
    prompt = f"""
Based on these AI predictions:
{json.dumps(session["predictions"], indent=2)}
And user responses:
{json.dumps(session["user_answers"], indent=2)}
Determine the final skin disease. Return ONLY the final disease name in plain text.
    """
    final_disease = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text.strip()

    treatment_prompt = f"""
You are a medical assistant. Provide a structured and easy-to-understand treatment plan for the following skin condition:

**Disease:** {final_disease}

Respond using this exact format. Each section should have **2–3 short bullet points**. Keep the explanations **simple, practical, and relevant for a general audience**.

**Diagnosis:** [Short explanation of the disease and how it's usually identified]

**Symptoms:**
• [Common symptom 1]
• [Common symptom 2]
• [Common symptom 3]

**Causes:**
• [Major cause or risk factor]
• [Another common contributing factor]

**Treatments (Ordered):**
• Ayurvedic Solutions: [1–2 natural treatments with brief benefits]
• Home Remedies: [1–2 things people can try at home for relief]
• Non-Prescription Medications: [1–2 OTC products with when to use them]
• Prescription Medications: [1–2 doctor-prescribed options and their purpose]

**When to See a Doctor:**
• [Early warning sign]
• [Progression or worsening symptom]
    """
    
    try:
        treatment_response = genai.GenerativeModel("gemini-2.0-flash").generate_content(treatment_prompt).text.strip()
    except Exception:
        treatment_response = "⚠️ Unable to fetch treatment details. Please consult a dermatologist."

    session["final_disease"] = final_disease
    return render_template(
        "final_diagnosis.html",
        final_disease=final_disease,
        treatment=treatment_response,
        detection_mode=session.get("detection_mode", "Text Only")
    )

@app.route("/treatment", methods=["GET"])
def treatment():
    final_disease = session.get("final_disease", "Unknown")
    treatment_prompt = f"Provide structured diagnosis and treatment for {final_disease} in a bullet-point format."
    treatment_plan = genai.GenerativeModel("gemini-2.0-flash").generate_content(treatment_prompt).text
    return render_template("treatment.html", final_disease=final_disease, treatment=treatment_plan)

# ------------------- Main Entry Point -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
