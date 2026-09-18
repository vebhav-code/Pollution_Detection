from fastapi import FastAPI, UploadFile, File, HTTPException

from services.gemini_service import detect_pollution


app = FastAPI(
    title="Pollution Detection API",
    version="1.0.0"
)


ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


@app.get("/")
def home():
    return {
        "message": "Pollution Detection API is running"
    }


@app.post("/detect-pollution")
async def pollution_detection(
    file: UploadFile = File(...)
):

    # Check file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are allowed"
        )

    # Read image
    image_bytes = await file.read()

    # Basic size limit: 10 MB
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image must be smaller than 10 MB"
        )

    try:

        result = await detect_pollution(
            image_bytes,
            file.content_type
        )

        return {
            "success": True,
            "filename": file.filename,
            "result": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Pollution detection failed: {str(e)}"
        )