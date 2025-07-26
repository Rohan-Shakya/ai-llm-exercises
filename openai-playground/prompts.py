import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = "gpt-4o"

# === Translation Prompt ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a translator."},
        {
            "role": "user",
            "content": """Translate these sentences: 
                'Hello' -> 'Hola', 
                'Goodbye' -> 'Adiós'. 
                Now translate: 'Thank you'.""",
        },
    ],
)
print("== Translation Response ==\n", response.choices[0].message.content, "\n")


# === Direct Q&A Prompt ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print("== Q&A Response ==\n", response.choices[0].message.content, "\n")


# === Chain of Thought Prompt ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a math tutor."},
        {
            "role": "user",
            "content": "Solve this math problem step by step: If John has 5 apples and gives 2 to Mary, how many does he have left?",
        },
    ],
)
print("== Math Tutor Response ==\n", response.choices[0].message.content, "\n")


# === Instructional Prompt ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a knowledgeable personal trainer and writer."},
        {"role": "user", "content": "Write a 300-word summary of the benefits of exercise, using bullet points."},
    ],
)
print("== Trainer Summary Response ==\n", response.choices[0].message.content, "\n")


# === Role-Playing Prompt ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a character in a fantasy novel."},
        {"role": "user", "content": "Describe the setting of the story."},
    ],
)
print("== Fantasy Roleplay Response ==\n", response.choices[0].message.content, "\n")


# === Open-Ended Philosophical Prompt ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a philosopher."},
        {"role": "user", "content": "What is the meaning of life?"},
    ],
)
print("== Philosopher Response ==\n", response.choices[0].message.content, "\n")


# === Creative Prompt with Sampling ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a creative tagline for a coffee shop."},
    ],
    temperature=0.9,
    top_p=0.9,
)
print("== Creative Tagline Response ==\n", response.choices[0].message.content, "\n")


# === Streaming Response Example ===
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a travel blogger."},
        {"role": "user", "content": "Write a 500-word blog post about your recent trip to Paris. Make sure to give a step-by-step itinerary of your trip."},
    ],
    temperature=0.9,
    stream=True,
)

print("== Travel Blog Streaming Response ==")
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
