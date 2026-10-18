
import os
import anthropic
from dotenv import load_dotenv, find_dotenv

path = find_dotenv()
print("Found .env at:", repr(path))
load_dotenv(path)
key = os.getenv("ANTHROPIC_API_KEY")
print("Co key:", bool(key), "| Do dai:", len(key) if key else 0)

client = anthropic.Anthropic(api_key=key)

try:
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Xin chao"}],
    )
    print("THANH CONG:", msg.content[0].text)
except Exception as e:
    print("Loi:", type(e).__name__)
    print(e)