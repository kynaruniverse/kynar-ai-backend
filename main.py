import os
import json
from zipfile import ZipFile
from fastapi import FastAPI, UploadFile
import openai
import math

# Set GEMINI_KEY in Render environment variables
openai.api_key = os.getenv("GEMINI_KEY")

app = FastAPI()

# Load previous embeddings if exist
if os.path.exists("repo_data.json"):
    with open("repo_data.json", "r") as f:
        repo_data = json.load(f)
else:
    repo_data = []

# --- Helper functions ---
def extract_text_files(zip_path):
    files_content = []
    with ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(('.ts', '.tsx', '.js', '.json')):
                content = zip_ref.read(file_name).decode(errors='ignore')
                files_content.append({"file_name": file_name, "content": content})
    return files_content

def create_embedding(text):
    response = openai.embeddings.create(
        model="gemini-embedding-001",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

# --- API Endpoints ---
@app.post("/upload-zip")
async def upload_zip(file: UploadFile):
    zip_path = f"temp_{file.filename}"
    with open(zip_path, "wb") as f:
        f.write(await file.read())
    
    files_content = extract_text_files(zip_path)
    
    for f in files_content:
        embedding = create_embedding(f["content"])
        repo_data.append({"file_name": f["file_name"], "content": f["content"], "embedding": embedding})
    
    with open("repo_data.json", "w") as f:
        json.dump(repo_data, f)
    
    return {"message": f"Indexed {len(files_content)} files successfully."}

@app.post("/ask")
async def ask_question(payload: dict):
    query = payload.get("query")
    if not query:
        return {"answer": "No query provided."}
    
    query_embedding = create_embedding(query)
    
    scores = [(cosine_similarity(query_embedding, f["embedding"]), f) for f in repo_data]
    scores.sort(reverse=True)
    top_files = [f["content"] for _, f in scores[:3]]  # Top 3 relevant files
    
    prompt = "Answer the following question based on these files:\n"
    for content in top_files:
        prompt += content + "\n"
    prompt += f"\nQuestion: {query}"
    
    response = openai.chat.completions.create(
        model="gemini-2.5-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    return {"answer": answer}