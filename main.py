# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import zipfile, io, os, json, math, asyncio
import openai
from typing import List, Dict

# ==== CONFIGURE ====
DATA_FILE = "repo_data.json"
GEMINI_KEY = os.getenv("GEMINI_KEY")  # Set in Render secrets
TOP_K_FILES = 3  # Number of top relevant files to send to Gemini

openai.api_key = GEMINI_KEY

app = FastAPI(title="Kynar AI Backend")

# ==== GLOBALS ====
repo_data: Dict[str, Dict] = {}

# ==== UTILITY FUNCTIONS ====
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

async def save_repo_data():
    """Save repo_data to JSON asynchronously."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: json.dump(repo_data, open(DATA_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2))

async def ensure_embedding(fname: str):
    """Compute embedding for a single file if missing."""
    if repo_data[fname]["embedding"] is None:
        emb = openai.embeddings.create(
            model="gemini-embedding-001",
            input=repo_data[fname]["text"]
        ).data[0].embedding
        repo_data[fname]["embedding"] = emb

# ==== LOAD EXISTING DATA ====
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        repo_data = json.load(f)
else:
    repo_data = {}

# ==== ROUTES ====
@app.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    """Upload a ZIP containing your repo files and store their text content."""
    try:
        content = await file.read()
        zip_bytes = io.BytesIO(content)

        updated_files = 0
        with zipfile.ZipFile(zip_bytes) as z:
            for f in z.namelist():
                if f.endswith((".ts", ".tsx", ".js", ".json", ".html", ".css")):
                    text = z.read(f).decode("utf-8", errors="ignore")
                    repo_data[f] = {"text": text, "embedding": None}
                    updated_files += 1

        await save_repo_data()

        return JSONResponse({
            "status": "success",
            "num_text_files": updated_files,
            "files": list(repo_data.keys())
        })

    except zipfile.BadZipFile:
        return JSONResponse({"status": "error", "message": "Not a valid ZIP"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/ask")
async def ask(query: dict):
    """Ask a question about the uploaded repo files using Gemini."""
    try:
        user_q = query.get("query")
        if not user_q:
            return JSONResponse({"status":"error","message":"No query provided"}, status_code=400)

        # Compute query embedding
        query_emb = openai.embeddings.create(
            model="gemini-embedding-001",
            input=user_q
        ).data[0].embedding

        # Ensure embeddings exist for all files concurrently
        await asyncio.gather(*(ensure_embedding(fname) for fname in repo_data.keys()))

        await save_repo_data()

        # Rank files by similarity
        scores = [(cosine_similarity(query_emb, data["embedding"]), fname) for fname, data in repo_data.items()]
        scores.sort(reverse=True)
        top_files = [repo_data[fname]["text"] for _, fname in scores[:TOP_K_FILES]]

        # Build prompt
        prompt_text = "Answer the question based on these files:\n"
        for i, t in enumerate(top_files):
            prompt_text += f"\n[File {i+1}]:\n{t}\n"
        prompt_text += f"\nQuestion: {user_q}"

        # Gemini chat response
        response = openai.chat.completions.create(
            model="gemini-2.5-chat",
            messages=[{"role":"user","content":prompt_text}]
        )
        answer = response.choices[0].message.content

        return JSONResponse({"status":"success","answer":answer})

    except Exception as e:
        return JSONResponse({"status":"error","message":str(e)}, status_code=500)