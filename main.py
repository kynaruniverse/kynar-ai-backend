# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import zipfile, io, os, json, math
import openai

# ==== CONFIGURE ====
DATA_FILE = "repo_data.json"
GEMINI_KEY = os.getenv("GEMINI_KEY")  # Set in Render secrets

openai.api_key = GEMINI_KEY

app = FastAPI()

# Utility: cosine similarity
def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

# Load repo_data.json if exists
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        repo_data = json.load(f)
else:
    repo_data = {}

# ==== STEP 1: Upload ZIP ====
@app.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    try:
        content = await file.read()
        zip_bytes = io.BytesIO(content)

        updated_files = 0
        with zipfile.ZipFile(zip_bytes) as z:
            for f in z.namelist():
                if f.endswith((".ts", ".tsx", ".js", ".json", ".html", ".css")):
                    text = z.read(f).decode("utf-8", errors="ignore")
                    repo_data[f] = {"text": text, "embedding": None}  # placeholder
                    updated_files += 1

        # Save to disk
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(repo_data, f, ensure_ascii=False, indent=2)

        return JSONResponse({
            "status": "success",
            "num_text_files": updated_files,
            "files": list(repo_data.keys())
        })

    except zipfile.BadZipFile:
        return JSONResponse({"status": "error", "message": "Not a valid ZIP"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# ==== STEP 2: Ask Gemini ====
@app.post("/ask")
async def ask(query: dict):
    try:
        user_q = query.get("query")
        if not user_q:
            return JSONResponse({"status":"error","message":"No query provided"}, status_code=400)

        # Compute embedding for query
        query_emb = openai.embeddings.create(
            model="gemini-embedding-001",
            input=user_q
        ).data[0].embedding

        # Ensure each file has embedding
        for fname, data in repo_data.items():
            if data["embedding"] is None:
                emb = openai.embeddings.create(
                    model="gemini-embedding-001",
                    input=data["text"]
                ).data[0].embedding
                repo_data[fname]["embedding"] = emb

        # Save updated embeddings
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(repo_data, f, ensure_ascii=False, indent=2)

        # Find top 3 relevant files
        scores = []
        for fname, data in repo_data.items():
            scores.append((cosine_similarity(query_emb, data["embedding"]), fname))
        scores.sort(reverse=True)
        top_files = [repo_data[fname]["text"] for _, fname in scores[:3]]

        # Create prompt
        prompt_text = "Answer the question based on these files:\n"
        for i, t in enumerate(top_files):
            prompt_text += f"\n[File {i+1}]:\n{t}\n"
        prompt_text += f"\nQuestion: {user_q}"

        # Call Gemini chat
        response = openai.chat.completions.create(
            model="gemini-2.5-chat",
            messages=[{"role":"user","content":prompt_text}]
        )
        answer = response.choices[0].message.content

        return JSONResponse({"status":"success","answer":answer})

    except Exception as e:
        return JSONResponse({"status":"error","message":str(e)}, status_code=500)