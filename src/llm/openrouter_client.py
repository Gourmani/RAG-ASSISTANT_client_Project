import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_llm():
    def generate(prompt):
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=512,
        )
        return response.choices[0].message.content

    return generate