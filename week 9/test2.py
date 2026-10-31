import os
import anthropic
from dotenv import load_dotenv , find_dotenv
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

load_dotenv(find_dotenv())
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5-5"

def get_text(msg) -> str:
    """Join all text blocks and skip thinking blocks."""
    return "".join(block.text for block in msg.content if block.type)

def log_usage(msg):
    print(f"[usage] input={msg.usage.input_tokens} output={msg.usage.output_tokens}")

def basic_call(prompt: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    log_usage(msg)
    return get_text(msg)

class Review(BaseModel):
    sentiment: Literal=["positive", "negative", "neutral"]
    score: int=Field(ge=1,le=10)
    summary: str

def structured_call(text: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=(
            "Return Only a JSON object with these keys: "
            "sentiment (positive, negative and neutral), "
            "score (integer from 1 to 10), "
            "summary (one short sentence). "
            "No markdown, no extra text. "
        ),
        messages=[{"role": "user", "content": text}]
    )
    log_usage(msg)
    raw = get_text(msg).strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()

    try:
        return Review.model_validate_json(raw)
    except ValidationError as e:
        print("JSON khong the hien thi:", e)
        return None


if __name__ == "__main__":

    try:
        print("--- Basic Call ---")
        print(basic_call("Xin chao hay gioi thieu doi chut ve ban"))
        review = structured_call("View dep nhung phuc vu hoi chap va cafe hoi nhat")

        if Review:
           print("Sentiment:", review.sentiment)
           print("Score:", review.score)
    except Exception as e:
        print("Loi:",type(e).__name__)
        print(e)
    

     
    

    


    