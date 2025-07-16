from transformers.pipelines import pipeline
from transformers import AutoTokenizer
from typing import Optional


def create_simple_llm(model_name: str = "distilgpt2"):
    """
    Initializes a simple causal language model using Hugging Face Transformers.

    Parameters
    ----------
    model_name : str
        Name of the pretrained model to use (default: 'distilgpt2').

    Returns
    -------
    generator : transformers.pipelines.text_generation.TextGenerationPipeline
        A ready-to-use text generation pipeline.
    """
    return pipeline(
        task="text-generation",
        model=model_name,
        pad_token_id=50256  # GPT-2-specific end-of-text token
    )


def generate_text(
    generator,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7
) -> str:
    """
    Generates a text continuation based on the input prompt.

    Parameters
    ----------
    generator : TextGenerationPipeline
        The pipeline returned by `create_simple_llm`.
    prompt : str
        Initial input text to seed the generation.
    max_new_tokens : int
        Number of new tokens to generate (after the prompt).
    temperature : float
        Sampling temperature to control randomness.

    Returns
    -------
    str
        Generated text output.
    """
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature
    )
    return result[0]["generated_text"]


def run_llm_demo():
    """
    Runs an educational demo with preset prompts and generated outputs.
    """
    print("🤖 Loading Simple LLM Model...")
    generator = create_simple_llm()

    print("\n✨ Simple LLM Demo ✨")
    print("This demonstration shows how a small LLM generates text from prompts.\n")

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Python programming is",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n🔹 Prompt {i}: {prompt}")
        output = generate_text(generator, prompt)
        print("🔸 Generated:", output)
        input("\n⏩ Press Enter to continue...")


def interactive_demo():
    """
    Allows the user to input custom prompts and get real-time completions.
    """
    generator = create_simple_llm()

    print("\n🧠 Interactive LLM Demo")
    print("Type your prompt below (type 'quit' to exit)\n")

    while True:
        prompt = input("✍️ Prompt: ").strip()
        if prompt.lower() == "quit":
            print("👋 Exiting interactive mode.")
            break
        if not prompt:
            print("⚠️ Empty prompt. Please try again.")
            continue

        output = generate_text(generator, prompt)
        print("\n💬 Generated Response:\n", output, "\n")


def explain_process():
    """
    Explains how tokenization and generation works with example input.
    """
    print("\n🔍 Understanding the LLM Process")
    print("1. Tokenization → Text → Tokens")
    print("2. Model Inference → Predict next tokens")
    print("3. Detokenization → Tokens → Text")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text = "Hello world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print("\n🧪 Tokenization Example:")
    print(f"Original Text : {text}")
    print(f"Token IDs     : {tokens}")
    print(f"Decoded Text  : {decoded}")


def main():
    """
    CLI interface for demo selection.
    """
    print("📚 Choose a Demo Mode:")
    print("1. Run Basic Demonstration")
    print("2. Interactive Prompting")
    print("3. Explain How It Works")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        run_llm_demo()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        explain_process()
    else:
        print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
