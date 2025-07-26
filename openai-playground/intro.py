import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an eastern poet."},
        {
            "role": "user",
            "content": """Write me a short poem about the moon. 
                          Write the poem in the style of a haiku.
                          Make sure to include a title for the poem.""",
        },
    ],
)

# Print the response content
print(response.choices[0].message.content)
