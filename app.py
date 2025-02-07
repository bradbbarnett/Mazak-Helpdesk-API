from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import openai
import os
import json

# Initialize Firestore with Service Account Key
firebase_creds = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API Key (Replace with your actual key)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def search_firestore(question):
    """Search Firestore for an existing answer"""
    docs = db.collection("qa_archive").stream()
    for doc in docs:
        data = doc.to_dict()
        if data["question"].lower() == question.lower():
            return data["answer"]
    return None

def generate_answer(question):
    """Use OpenAI GPT-3.5-turbo to generate an answer"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use an available model
        messages=[{"role": "user", "content": question}]
    )

    return response.choices[0].message.content

@app.route("/get_answer", methods=["POST"])
def get_answer():
    """API Endpoint for retrieving or generating answers"""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Check Firestore first
    existing_answer = search_firestore(question)
    if existing_answer:
        return jsonify({"answer": existing_answer, "source": "database"})

    # Generate a new answer using OpenAI
    new_answer = generate_answer(question)

    # Store the new Q&A pair in Firestore
    db.collection("qa_archive").add({
        "question": question,
        "answer": new_answer
    })

    return jsonify({"answer": new_answer, "source": "AI-generated"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
