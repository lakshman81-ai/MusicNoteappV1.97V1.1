from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os
import tempfile
from typing import Optional

from backend.transcription import transcribe_audio_pipeline, transcribe_audio

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    stereo_mode: bool = Form(False),
    start_offset: float = Form(0.0),
    max_duration: Optional[float] = Form(None),
):
    """Endpoint to handle audio file upload and return MusicXML."""

    try:
        suffix = os.path.splitext(file.filename or "upload")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        try:
            result = transcribe_audio_pipeline(
                tmp_path,
                stereo_mode=stereo_mode,
                start_offset=start_offset,
                max_duration=max_duration,
            )
            xml_bytes = result["musicxml"].encode("utf-8")
            return Response(content=xml_bytes, media_type="application/xml")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
