from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

# Initialize chat model (using OpenAI GPT-3.5)
model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# print(messages)

# response = model.invoke(messages)

# print(response)


## Langchain promptTemplates: https://python.langchain.com/docs/tutorials/llm_chain/#prompt-templates
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "French", "text": "Bonjour!"})

# print(prompt)

response = model.invoke(prompt)
print(response.content)