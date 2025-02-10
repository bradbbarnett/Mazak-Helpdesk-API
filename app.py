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
    """Search Firestore for similar questions using embeddings."""
    query_embedding = get_embedding(question)
    best_match = None
    best_score = 0.0  # Cosine similarity (1.0 = perfect match)

    docs = db.collection("qa_archive").stream()
    for doc in docs:
        data = doc.to_dict()
        stored_embedding = data.get("embedding")

        if stored_embedding:
            similarity = 1 - cosine(query_embedding, stored_embedding)
            if similarity > best_score:  # Find the highest similarity
                best_score = similarity
                best_match = data

    if best_match and best_score >= 0.85:  # Confidence threshold
        return best_match["answer"], best_score * 100  # Convert to percentage

    return None, best_score * 100


def generate_answer(question):
    """Use OpenAI to generate an email response."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Write a professional email response: {question}"}]
    )
    return response.choices[0].message.content


@app.route("/get_answer", methods=["POST"])
def get_answer():
    """API Endpoint for retrieving or generating answers."""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Search Firestore for similar questions
    existing_answer, confidence = search_firestore(question)
    if existing_answer:
        return jsonify({
            "answer": existing_answer,
            "source": "database",
            "confidence": confidence
        })

    # No match found, generate a new response
    new_answer = generate_answer(question)

    return jsonify({
        "answer": new_answer,
        "source": "AI-generated",
        "confidence": confidence,  # Confidence remains low if no match
        "note": "This answer is AI-generated and not stored in the database."
    })


@app.route("/confirm_answer", methods=["POST"])
def confirm_answer():
    """API endpoint to confirm and store a correct answer in Firestore."""
    data = request.json
    question = data.get("question")
    answer = data.get("answer")

    if not question or not answer:
        return jsonify({"error": "Question and answer required"}), 400

    embedding = get_embedding(question)  # Store embedding for future searches

    db.collection("qa_archive").add({
        "question": question,
        "answer": answer,
        "embedding": embedding
    })

    return jsonify({"message": "Answer confirmed and stored in Firestore."})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
