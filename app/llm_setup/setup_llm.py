from openai import OpenAI
# load_dotenv()  # Loads your .env file into os.environ

def set_up_llm():
    client = OpenAI()
    return client


def prompt_llm(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
