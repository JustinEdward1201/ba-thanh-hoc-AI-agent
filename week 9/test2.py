import os
import anthropic
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv(find_dotenv())
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5-5"


def get_text(msg) -> str:
    """Join all text blocks and skip thinking blocks."""
    return "".join(block.text for block in msg.content if block.type == "text")


def log_usage(msg):
    print(f"[usage] input={msg.usage.input_tokens} output={msg.usage.output_tokens}")


# ---------- Part 1: basic call ----------
def basic_call(prompt: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    log_usage(msg)
    return get_text(msg)


# ---------- Part 2: structured output (JSON + Pydantic) ----------
class Review(BaseModel):
    sentiment: str   # positive / negative / neutral
    score: int       # 1-10
    summary: str


def structured_call(text: str):
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=(
            "Return ONLY a JSON object with these keys: "
            "sentiment (positive, negative or neutral), "
            "score (integer from 1 to 10), "
            "summary (one short sentence). "
            "No markdown, no extra text."
        ),
        messages=[{"role": "user", "content": text}],
    )
    log_usage(msg)

    raw = get_text(msg).strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()

    try:
        return Review.model_validate_json(raw)
    except ValidationError as e:
        print("JSON khong hop le:", e)
        return None


if __name__ == "__main__":
    try:
        print("--- Basic call ---")
        print(basic_call("Xin chao, gioi thieu ngan ve ban."))

        print("\n--- Structured output ---")
        review = structured_call("Quan ca phe nay dep, nhung phuc vu cham va ca phe hoi nhat.")
        if review:
            print(review)
            print("Sentiment:", review.sentiment)
            print("Score:", review.score)
    except Exception as e:
        print("Loi:", type(e).__name__)
        print(e)