import os
import anthropic
from dotenv import load_dotenv, find_dotenv
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

load_dotenv(find_dotenv())
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5-5"
PRICE_IN = 2.00
PRICE_OUT = 10.00
total_cost = 0.00

def get_text(msg) -> str:
    """Join all text blocks and skip thinking blocks."""
    return "".join(block.text for block in msg.content if block.type == "text")

def log_usage(msg):
    global total_cost #chay bien ngoai ma khong anh huong
    cost = calc_cost(msg.usage)
    print(f"[usage] input={msg.usage.input_tokens} output={msg.usage.output_tokens} cost: ${cost: .6f}")

def calc_cost(usage) -> float:
    '''Tinh chi phi (USD) tu so token in va out moi lan goi'''
    cost_in = usage.input_tokens / 1_000_000 * PRICE_IN
    cost_out = usage.input_tokens / 1_000_000 * PRICE_OUT
    return cost_in + cost_out

def basic_call(prompt: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens = 2000,
        messages=[{"role": "user", "content": prompt}],
    )
    log_usage(msg)
    return get_text(msg)

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: int=Field(ge=1,le=10)
    summary: str

def structured_call(text: str) -> Review | None:
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
        messages = [{"role": "user", "content": text}],
    )
    log_usage(msg)
    raw = get_text(msg).strip()
    raw = raw.removeprefix("```json").removesuffix("```").strip()

if __name__ == "__main__":
    try:
        print("--- Basic Call ---")
        print(basic_call("Xin chao, hay gioi thieu doi chut ve ban"))
        review = structured_call("View dep nhung phuc vu hoi cham va cafe hoi nhat")

        if Review:
            print("Sentiment:", review.sentiment)
            print("Score:", review.score)
        print(f"\nTong chi phi call la cost:${total_cost: .6f}")
    except Exception as e:
        print("Loi:",type(e).__name__)
        print(e)