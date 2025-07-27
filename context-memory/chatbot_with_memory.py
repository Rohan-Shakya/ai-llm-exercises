import json
import sys
from typing import List, cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

load_dotenv()

MODEL_OPENAI = "gpt-4o"
MODEL_OLAMMA = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
CONVERSATION_FILE = "conversation.json"


def initialize_client(use_ollama: bool = False) -> OpenAI:
    """Initialize the OpenAI client for either OpenAI or OLLAMA."""
    if use_ollama:
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return OpenAI()


def create_initial_messages() -> List[ChatCompletionMessageParam]:
    """Create the initial system message for a new conversation."""
    return [{"role": "system", "content": "You are a helpful assistant."}]


def chat(
    user_input: str,
    messages: List[ChatCompletionMessageParam],
    client: OpenAI,
    model_name: str,
) -> str:
    """Handle user input and generate assistant responses."""
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        print("Full API Response:", response.model_dump_json(indent=2))

        if response.choices:
            content = response.choices[0].message.content
            if content is None:
                return "Error: Assistant returned empty content."
            messages.append({"role": "assistant", "content": content})
            return content
        else:
            return "Error: No choices in the response."

    except Exception as e:
        return f"Error with API: {str(e)}"


def summarize_messages(messages: List[ChatCompletionMessageParam]) -> List[ChatCompletionMessageParam]:
    """Summarize previous conversation to preserve tokens."""
    summary_parts = []

    for m in messages[-5:]:
        if m["role"] == "system":
            continue
        content = m.get("content")
        if isinstance(content, str):
            summary_parts.append(content[:50] + "...")
        else:
            summary_parts.append("[non-text message]")

    summary = "Previous conversation summarized: " + " ".join(summary_parts)

    # Cast to resolve type mismatch caused by list addition
    summarized: List[ChatCompletionMessageParam] = cast(
        List[ChatCompletionMessageParam],
        [{"role": "system", "content": summary}] + messages[-5:]
    )

    return summarized


def save_conversation(messages: List[ChatCompletionMessageParam], filename: str = CONVERSATION_FILE):
    """Persist conversation to a local JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(messages, f)
        print("Conversation saved.")
    except IOError as e:
        print(f"Error saving conversation: {str(e)}")


def load_conversation(filename: str = CONVERSATION_FILE) -> List[ChatCompletionMessageParam]:
    """Load a previously saved conversation."""
    try:
        with open(filename, "r") as f:
            loaded = json.load(f)
            return cast(List[ChatCompletionMessageParam], loaded)
    except FileNotFoundError:
        print("No conversation file found. Starting a new one.")
        return create_initial_messages()
    except json.JSONDecodeError:
        print("Error decoding conversation file. Starting fresh.")
        return create_initial_messages()


def display_commands():
    """Display available user commands."""
    print("\nAvailable commands:")
    print("- 'save': Save conversation")
    print("- 'load': Load conversation")
    print("- 'summary': Summarize conversation")
    print("- 'quit': Exit the chat")


def main():
    # Model selection
    print("Select model type:")
    print("1. OpenAI GPT-4")
    print("2. Ollama (Local)")

    choice = input("Enter choice (1 or 2): ").strip()
    use_ollama = choice == "2"

    # Initialize client and model
    client = initialize_client(use_ollama)
    model_name = MODEL_OLAMMA if use_ollama else MODEL_OPENAI

    # Load or start new conversation
    messages = load_conversation()

    print(f"\nUsing {'Ollama' if use_ollama else 'OpenAI'} model.")
    display_commands()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "save":
            save_conversation(messages)
            continue
        elif user_input.lower() == "load":
            messages = load_conversation()
            continue
        elif user_input.lower() == "summary":
            messages = summarize_messages(messages)
            print("Conversation summarized.")
            continue

        response = chat(user_input, messages, client, model_name)
        print(f"\nAssistant: {response}")

        if len(messages) > 20:
            messages = summarize_messages(messages)
            print("\n(Note: Conversation auto-summarized to save memory.)")


if __name__ == "__main__":
    main()
