import io
import os
import json
import pandas as pd
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import tiktoken  # for token counting
from collections import defaultdict

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

# Initialize OpenAI client
client = OpenAI()

# Set up the tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Helper function to convert JSON to JSONL format
def json_to_jsonl(input_file, output_file):
    """
    Converts a JSON file to JSONL format.
    
    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSONL file.
    """
    with open(input_file) as f:
        data = json.load(f)

    with open(output_file, "w") as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write("\n")

# Helper function to check the format of the dataset
def check_file_format(dataset):
    """
    Validates the format of the dataset.
    
    :param dataset: The dataset to validate (a list of JSON entries).
    """
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

# Helper function to count tokens from messages
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    """
    Calculates the number of tokens in the given messages.
    
    :param messages: The messages to calculate tokens for.
    :param tokens_per_message: The number of tokens per message.
    :param tokens_per_name: The number of tokens per name (optional).
    :return: The total number of tokens.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # Add additional tokens for the conversation
    return num_tokens

# Convert JSON to JSONL
json_to_jsonl("teacrafter.json", "output.jsonl")

# Load the dataset from the JSONL file
data_path = "output.jsonl"
with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

# Print dataset stats and first example
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)

# Validate dataset format
check_file_format(dataset)

# Estimate token usage for the dataset
conversation_length = []
for msg in dataset:
    messages = msg["messages"]
    conversation_length.append(num_tokens_from_messages(messages))

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096
TARGET_EPOCHS = 5
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)

# Adjust n_epochs based on dataset size
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

# Calculate total billing tokens
n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in conversation_length)
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

num_tokens = n_epochs * n_billing_tokens_in_dataset

# Cost calculation (for gpt-3.5-turbo pricing)
cost = (num_tokens / 1000) * 0.0080  # Update pricing based on the model used
print(cost)

# Fine-tune the model (example: for gpt-4o-mini)
# Training file upload and fine-tuning process (currently commented out)
# training_file = client.files.create(
#     file=open("output.jsonl", "rb"), purpose="fine-tune"
# )
# print(training_file.id)

# Start a fine-tuning job
job_id = "ftjob-dCBMHb7AtBdqWrgMG9KZqdfaseI5y"
state = client.fine_tuning.jobs.retrieve(job_id)
# print(f"Fine-tuning job status: {state}")

# Retrieve the results of the fine-tuned model
result_file = "file-6tZRoEV4SJ8fwjuWuPQpszzwYS"
file_data = client.files.content(result_file)
file_data_bytes = file_data.read()
file_like_object = io.BytesIO(file_data_bytes)

# Read CSV file to create a DataFrame
df = pd.read_csv(file_like_object)
# print(df)

# Example to test the fine-tuned model with the dataset
fine_tuned_model = "ft:gpt-4o-2024-08-06:personal::Bd3Auqao2-getyourowd"
response = client.chat.completions.create(
    model=fine_tuned_model,
    messages=[
        {"role": "system", "content": "This is a customer support chatbot designed to help with common inquiries."},
        {"role": "user", "content": "How do I change my tea preferences for the next shipment?"}
    ]
)
print(f"Fine-tuned model response: \n{response.choices[0].message.content}")

# Collect messages from the user and assistant
context = [{"role": "system", "content": "This is a customer support chatbot designed to help with common inquiries for TeaCrafters"}]

def collect_messages(role, message):
    """Collects messages exchanged between the user and assistant."""
    context.append({"role": role, "content": message})

def get_completion():
    """Generates a completion using the fine-tuned model."""
    try:
        response = client.chat.completions.create(model=fine_tuned_model, messages=context)
        print("\n Assistant: ", response.choices[0].message.content, "\n")
        return response.choices[0].message.content
    except openai.APIError as e:
        print(e.http_status)
        print(e.error)
        return e.error

# Start the conversation with the assistant
while True:
    collect_messages("assistant", get_completion())  # stores assistant's response
    user_prompt = input("User: ")  # get input from the user

    if user_prompt == "exit":  # end the conversation if the user types "exit"
        print("\n Goodbye")
        break

    collect_messages("user", user_prompt)  # stores user's prompt
