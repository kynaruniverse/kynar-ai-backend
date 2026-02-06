from fastapi import Body
import openai
import math

# Replace with your Gemini key (or use environment variable)
openai.api_key = "YOUR_GEMINI_KEY"

# Load repo_data.json
def load_repo():
    import json
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Save repo_data.json
def save_repo(repo_data):
    import json
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(repo_data, f, ensure_ascii=False, indent=2)

# Compute cosine similarity
def cosine_similarity(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

# Endpoint to ask questions
@app.post("/ask")
async def ask_question(payload: dict = Body(...)):
    query = payload.get("query")
    if not query:
        return JSONResponse({"status": "error", "message": "No query provided"}, status_code=400)

    repo_data = load_repo()

    # Step 1: compute query embedding
    query_embedding = openai.embeddings.create(
        model="gemini-embedding-001",
        input=query
    ).data[0].embedding

    # Step 2: compute similarity with each file
    scores = []
    for fname, content in repo_data.items():
        # compute embedding if not exists
        if not isinstance(content, dict):  # plain text
            emb = openai.embeddings.create(
                model="gemini-embedding-001",
                input=content
            ).data[0].embedding
            repo_data[fname] = {"text": content, "embedding": emb}
        else:
            emb = content["embedding"]

        sim = cosine_similarity(query_embedding, emb)
        scores.append((sim, fname))

    # Step 3: pick top 3 files
    top_files = sorted(scores, reverse=True)[:3]

    # Step 4: prepare prompt
    prompt_text = ""
    for _, fname in top_files:
        prompt_text += f"File: {fname}\n{repo_data[fname]['text']}\n\n"

    prompt_text += f"Question: {query}"

    # Step 5: call Gemini chat
    response = openai.chat.completions.create(
        model="gemini-2.5-chat",
        messages=[{"role": "user", "content": prompt_text}]
    )

    answer = response.choices[0].message.content

    # save updated embeddings
    save_repo(repo_data)

    return JSONResponse({"answer": answer})