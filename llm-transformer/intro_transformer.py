from transformers.pipelines import pipeline
from typing import List


def create_simple_llm(model_name: str = "distilgpt2"):
    """
    Initializes a simple text generation pipeline using a lightweight GPT-2 model.

    Parameters
    ----------
    model_name : str
        Pretrained model identifier from Hugging Face model hub.
        Default is 'distilgpt2', a small and efficient GPT-2 variant (~82M parameters).

    Returns
    -------
    generator : transformers.pipelines.text_generation.TextGenerationPipeline
        A pipeline for generating text using causal language modeling.

    Notes
    -----
    - distilgpt2 is designed for low-latency CPU or GPU inference.
    - pad_token_id=50256 avoids padding errors with GPT-2 models.
    """
    return pipeline(
        task="text-generation",
        model=model_name,
        pad_token_id=50256  # End-of-text token ID used as a padding substitute
    )


def generate_text(
    prompt: str,
    max_new_tokens: int = 256,
    num_return_sequences: int = 1,
    temperature: float = 0.7
) -> List[str]:
    """
    Generates text from a given prompt using the LLM.

    Parameters
    ----------
    prompt : str
        Initial text input to seed the generation.
    max_new_tokens : int
        Number of tokens to generate *after* the prompt.
    num_return_sequences : int
        Number of unique sequences to return.
    temperature : float
        Controls randomness in sampling (0 = deterministic, >1 = more creative).

    Returns
    -------
    List[str]
        A list of generated text continuations.
    """
    generator = create_simple_llm()

    results = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=temperature
    )

    return [result["generated_text"] for result in results]


if __name__ == "__main__":
    input_prompt = "Once upon a time"
    outputs = generate_text(input_prompt, max_new_tokens=256)

    for idx, text in enumerate(outputs, 1):
        print(f"\n--- Generated Text {idx} ---\n{text}\n")
