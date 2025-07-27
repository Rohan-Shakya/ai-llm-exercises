from openai import OpenAI
import sys

MODEL_GPT4 = "gpt-4o"
MODEL_OLAMMA = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1/"

ERROR_MSG = "An error occurred while processing your request."

def get_openai_client(use_ollama: bool) -> OpenAI:
    """
    Returns an OpenAI client instance based on whether Ollama or OpenAI is chosen.
    """
    if use_ollama:
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    else:
        return OpenAI()


def get_model_name(use_ollama: bool) -> str:
    """
    Returns the model name based on the selected option (Ollama or OpenAI).
    """
    return MODEL_OLAMMA if use_ollama else MODEL_GPT4


def simple_chat_without_memory(client: OpenAI, model_name: str, user_input: str) -> str:
    """
    Simple chat function that handles a single user input and returns the bot's response.
    Does not keep memory of previous inputs.
    """
    try:
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": user_input}]
        )
        return response.choices[0].message.content or "No response from the bot."
    except Exception as e:
        return f"{ERROR_MSG} Error: {str(e)}"



def print_welcome_message():
    """
    Prints the initial messages for the user.
    """
    print("\n=== Simple Chatbot WITHOUT Memory ===")
    print("Notice how the bot won't remember anything from previous messages!")
    print("\nSelect model type:")
    print("1. OpenAI GPT-4")
    print("2. Ollama (Local)")


def print_exit_or_clear_instructions():
    """
    Prints instructions to exit or clear the screen.
    """
    print("\n=== Chat Session Started ===")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear the screen")
    print("Each message is independent - the bot has no memory of previous messages!\n")


def handle_user_input():
    """
    Handles user input, checking for special commands like quit, exit, or clear.
    Returns the cleaned input string or None if it's an exit command.
    """
    user_input = input("\nYou: ").strip()

    # Check for exit commands
    if user_input.lower() in ["quit", "exit"]:
        print("\nGoodbye! 👋")
        sys.exit()

    # Check for clear command
    if user_input.lower() == "clear":
        print("\033[H\033[J", end="")
        return None

    # Skip empty inputs
    if not user_input:
        return None

    return user_input


def main():
    print_welcome_message()

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Please enter either 1 or 2")

    use_ollama = choice == "2"
    client = get_openai_client(use_ollama)
    model_name = get_model_name(use_ollama)

    print_exit_or_clear_instructions()

    while True:
        user_input = handle_user_input()

        if user_input is None:
            continue

        response = simple_chat_without_memory(client, model_name, user_input)
        print(f"\nBot: {response}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChat session ended by user. Goodbye! 👋")
        sys.exit()
