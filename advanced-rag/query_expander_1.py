import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


class QueryExpander:
    """
    Expands a natural language query into semantically diverse variations
    to enhance retrieval performance in search or knowledge systems.
    """

    _PROMPT_TEMPLATE = """Given the following question, generate 3 different versions of the question 
    that capture different aspects and perspectives of the original question. 
    Make the variations semantically diverse but relevant.
    
    Original Question: {question}
    
    Generate variations in the following format:
    1. [First variation]
    2. [Second variation]
    3. [Third variation]
    
    Only output the numbered variations, nothing else."""

    def __init__(self, temperature: float = 0.3):
        """
        Initializes the QueryExpander.

        Args:
            temperature (float): Degree of randomness in LLM output. Lower values produce more focused results.
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-4o-mini")

        self.query_expansion_prompt = PromptTemplate(
            input_variables=["question"],
            template=self._PROMPT_TEMPLATE,
        )

    def expand_query(self, question: str) -> List[str]:
        """
        Expands a single query into semantically rich alternatives.

        Args:
            question (str): Original user query.

        Returns:
            List[str]: List of query variations (original + 3 expanded).
        """
        try:
            response = self.llm.invoke(
                self.query_expansion_prompt.format(question=question)
            )

            # Parse the 3 numbered variations
            lines = response.content.strip().split("\n")
            variations = [
                line.split(". ", 1)[1].strip()
                for line in lines
                if line.strip() and ". " in line
            ]

            if question not in variations:
                variations.append(question)

            return variations

        except Exception as e:
            print(f"[Error] Query expansion failed: {repr(e)}")
            return [question]


def main():
    """
    Demonstrates usage of the QueryExpander class with example queries.
    """
    expander = QueryExpander()

    questions = [
        "What are the main causes of global warming?",
        "How does exercise affect mental health?",
        "What are the benefits of renewable energy?",
    ]

    for original in questions:
        print(f"\nOriginal Question: {original}")
        print("Expanded Queries:")

        for i, query in enumerate(expander.expand_query(original), 1):
            print(f"{i}. {query}")


if __name__ == "__main__":
    main()
