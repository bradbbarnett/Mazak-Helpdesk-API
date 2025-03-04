from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import openai
import os
import json
import numpy as np
from scipy.spatial.distance import cosine

# Initialize Firestore with Service Account Key
firebase_creds = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API Key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding Model
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text):
    """Generate an embedding for a given text using OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def search_firestore(question):
    """Search Firestore for relevant facts based on keywords and embeddings."""
    
    # Extract keywords (can later improve with NLP)
    keywords = question.lower().split()  # Simple keyword extraction

    # Try direct match in Firestore
    docs = db.collection("fact_database").stream()
    best_fact = None
    best_match_score = 0

    for doc in docs:
        data = doc.to_dict()
        fact_keywords = data.get("keywords", [])

        # Count how many keywords match
        match_score = sum(1 for word in keywords if word in fact_keywords)

        if match_score > best_match_score:
            best_match_score = match_score
            best_fact = data.get("fact")

    if best_fact and best_match_score > 0:  # Ensure some relevance
        return best_fact, min(100, best_match_score * 20)  # Scale confidence

    return None, 0

def generate_answer(question):
    """Use OpenAI to generate an email response."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Write a professional email response: {question}"}]
    )
    return response.choices[0].message.content

@app.route("/get_answer", methods=["POST"])
def get_answer():
    """Retrieve answers from facts or generate an AI response."""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # First try to find a relevant fact
    existing_fact, confidence = search_firestore(question)

    if existing_fact:
        return jsonify({
            "answer": f"Dear Customer,\n\n{existing_fact}\n\nBest regards,\nYour Support Team",
            "source": "Fact-based database",
            "confidence": confidence
        })

    # No fact found, generate AI-based response
    new_answer = generate_answer(question)

    return jsonify({
        "answer": f"Dear Customer,\n\n{new_answer}\n\nBest regards,\nYour Support Team",
        "source": "AI-generated",
        "confidence": 0,
        "note": "This answer is AI-generated and not stored in the database."
    })


    # No match found, generate a new response
    new_answer = generate_answer(question)

    return jsonify({
        "answer": new_answer,
        "source": "AI-generated",
        "confidence": 0,  # Confidence of 0 = Not found in Firestore
        "note": "This answer is AI-generated and not stored in the database."
    })

@app.route("/confirm_answer", methods=["POST"])
def confirm_answer():
    """Store a verified fact in the Firestore database."""
    data = request.json
    fact = data.get("fact")
    category = data.get("category", "General")
    keywords = data.get("keywords", [])

    if not fact or not keywords:
        return jsonify({"error": "Fact and keywords are required"}), 400

    db.collection("fact_database").add({
        "category": category,
        "keywords": keywords,
        "fact": fact
    })

    return jsonify({"message": "Fact stored successfully."})

if __name__ == "__main__":
    app.run(port=5000, debug=True)