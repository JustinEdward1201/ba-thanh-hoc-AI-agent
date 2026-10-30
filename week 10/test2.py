import os
from typing import Literal
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5-5"

class DanhGia(BaseModel):
    cam_xuc: Literal["tich cuc", "tieu cuc", "trung tinh"]
    diem: Field(ge=1, le=10)
    tom_tat: str

SYSTEM_PROMPT = """..."""


MAX_LAN_THU = 3

def phan_tich_co_thu_lai(danh_gia: str) -> DanhGia | None:
    messages = [{"role": "user", "content": danh_gia}]

    for lan in range (1, MAX_LAN_THU +1):
        print(f"Lan thu {lan}...")
        message = client.messages.create(
            model=MODEL,
            max_tokens = 300,
            system = SYSTEM_PROMPT,
            messages=messages,
        )
        text = message.content[0].text

        try:
            return DanhGia.model_validate_json(text)
        
        except ValidationError as e:
            print("  Sai:", e.error_count(), "Loi")

            message.append({"role": "assitant", "content": text})
            message.append({"role": "user", "content": ...})
        
    print("Da thu het", MAX_LAN_THU, "lan ma van sai.")
    return None