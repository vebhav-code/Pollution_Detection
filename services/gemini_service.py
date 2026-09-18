import os
import json
import base64

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


PROMPT = """
Analyze the image for visible environmental pollution.

Categories:
Garbage/Waste, Crop Burning, Air, Water, Plastic, Industrial, Sewage, Deforestation,Vehicle pollution.

Return ONLY valid JSON:
{
  "pollution_detected": true,
  "pollution_types": [],
  "confidence": "high",
  "description": "",
  "evidence": []
}

Use only visually observable evidence. Do not guess.
If no pollution is visible, use "No Pollution Detected".
If unclear, use "Unknown/Unclear".
"""


async def detect_pollution(
    image_bytes: bytes,
    mime_type: str
):

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=[
            {
                "type": "text",
                "text": PROMPT
            },
            {
                "type": "image",
                "data": image_base64,
                "mime_type": mime_type
            }
        ]
    )

    text = interaction.output_text.strip()

    # Remove markdown code fences if Gemini returns them
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)