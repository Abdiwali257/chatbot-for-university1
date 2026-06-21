import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

key = os.getenv("GROQ_API_KEY")

if not key:
    raise RuntimeError("GROQ_API_KEY was not found. Please check your .env file.")

client = Groq(api_key=key)

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    messages=[
        {"role": "user", "content": "Say only: Groq API is working."}
    ],
    temperature=0.2,
    max_tokens=50
)

print(response.choices[0].message.content)
