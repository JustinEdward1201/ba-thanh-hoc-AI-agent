import os
from typing import Literal

import anthropic
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv(find_dotenv())
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5-5"


def get_text(msg) -> str:
    """Join all text blocks and skip thinking blocks."""
    # Duyet tung block, chi giu block co type == "text", roi noi lai
    return "".join(block.text for block in msg.content if block.type == "text")


def log_usage(msg):
    print(f"[usage] input={msg.usage.input_tokens} output={msg.usage.output_tokens}")


def basic_call(prompt: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    log_usage(msg)
    return get_text(msg)


class Review(BaseModel):
    # SUA: Literal phai co danh sach gia tri trong [...]
    # Dau ":" la kieu du lieu, dau "=" la gia tri mac dinh
    sentiment: Literal["positive", "negative", "neutral"]
    score: int = Field(ge=1, le=10)
    summary: str


# SUA: ham tra ve Review hoac None, khong phai str
def structured_call(text: str) -> Review | None:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        # SUA: "or" thay cho "and", moi dong ket thuc bang khoang trang
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
        print(basic_call("Xin chao, hay gioi thieu ngan ve ban."))

        # SUA: them \n de tach dong voi phan tren
        print("\n--- Structured output ---")
        review = structured_call("Quan cafe nay dep nhung phuc vu cham va cafe hoi nhat.")

        if review:
            print(review)
            print("Sentiment:", review.sentiment)
            print("Score:", review.score)
    except Exception as e:
        print("Loi:", type(e).__name__)
        print(e)