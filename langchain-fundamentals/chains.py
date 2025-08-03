from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Step 1: Define the prompt template
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

# Step 2: Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")

# Step 3: Chain together the prompt, model, and output parser
chain = prompt | model | StrOutputParser()

# Step 4: Run the chain with input
if __name__ == "__main__":
    response = chain.invoke({"topic": "Nepal"})
    print(response)
