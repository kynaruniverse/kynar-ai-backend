# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import zipfile
import io
import os

app = FastAPI()

@app.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    try:
        # Read ZIP bytes
        content = await file.read()
        zip_bytes = io.BytesIO(content)

        # Extract file names (text files only)
        file_list = []
        with zipfile.ZipFile(zip_bytes) as z:
            for f in z.namelist():
                if f.endswith((".ts", ".tsx", ".js", ".json", ".html", ".css")):
                    file_list.append(f)

        return JSONResponse({
            "status": "success",
            "num_text_files": len(file_list),
            "files": file_list
        })

    except zipfile.BadZipFile:
        return JSONResponse({"status": "error", "message": "Not a valid ZIP"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)