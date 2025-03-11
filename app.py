from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import openai
from pinecone import Pinecone, ServerlessSpec
import os
import json
import logging

# 🔹 Initialize Firestore with Service Account Key
firebase_creds = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# 🔹 Initialize Flask app
app = Flask(__name__)

# 🔹 Set OpenAI API Key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔹 Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get Pinecone index name and ensure it's lowercase
index_name = os.getenv("PINECONE_INDEX_NAME", "").lower()

# Ensure the index exists, create if necessary
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust based on your embedding size
        metric="euclidean",
        spec=ServerlessSpec(
            cloud="aws",  # Update based on your Pinecone setup
            region=os.getenv("PINECONE_ENVIRONMENT")  # Uses your environment variable
        )
    )

# Connect to the index
index = pc.Index(index_name)

# 🔹 Embedding Model
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text):
    """Generate an embedding for a given text using OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def store_embeddings():
    """Generate and store embeddings for existing Firestore questions."""
    docs = db.collection("fact_database").stream()  # Ensure using the correct collection
    for doc in docs:
        data = doc.to_dict()
        question = data.get("question")
        answer = data.get("fact")  # Ensure it's using the correct field

        if not question or not answer:
            continue  # Skip invalid data

        embedding = get_embedding(question)

        # Store in Pinecone
        index.upsert([
            (doc.id, embedding, {"question": question, "answer": answer})
        ])
    
    print("✅ All embeddings stored in Pinecone!")

def search_faq(query):
    """Search for the best-matching fact using vector similarity."""
    query_embedding = get_embedding(query)

    # Search for closest matches
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    if results["matches"]:
        best_match = results["matches"][0]
        similarity = best_match["score"]

        if similarity > 0.85:  # High-confidence threshold
            return best_match["metadata"]["answer"], similarity
        else:
            return None, similarity  # Not confident enough

    return None, 0  # No match found

def generate_answer(question):
    """Use OpenAI to generate an AI response."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Write a professional email response: {question}"}]
    )
    return response.choices[0].message.content
    
    logging.basicConfig(level=logging.DEBUG)

@app.route("/get_answer", methods=["POST"])
def get_answer():
    """Retrieve answers from vector search or generate an AI response."""
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 🔹 First, search the vector database (Pinecone)
    existing_fact, confidence = search_faq(question)

    if existing_fact:
        return jsonify({
            "answer": f"Dear Customer,\n\n{existing_fact}\n\nBest regards,\nYour Support Team",
            "source": "Fact-based database",
            "confidence": round(confidence, 2)  # Confidence score rounded
        })

    # 🔹 If no fact found, generate AI-based response
    new_answer = generate_answer(question)

    return jsonify({
        "answer": f"Dear Customer,\n\n{new_answer}\n\nBest regards,\nYour Support Team",
        "source": "AI-generated",
        "confidence": 0,
        "note": "This answer is AI-generated and not stored in the database."
    })

@app.route("/confirm_answer", methods=["POST"])
def confirm_answer():
    """Store a verified fact in Firestore and Pinecone."""
    data = request.json
    question = data.get("question")
    fact = data.get("fact")
    category = data.get("category", "General")

    if not question or not fact:
        return jsonify({"error": "Both question and fact are required"}), 400

    # 🔹 Save in Firestore
    doc_ref = db.collection("fact_database").add({
        "category": category,
        "question": question,
        "fact": fact
    })

    # 🔹 Save in Pinecone
    embedding = get_embedding(question)
    index.upsert([(doc_ref[1].id, embedding, {"question": question, "answer": fact})])

    return jsonify({"message": "Fact stored successfully."})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
