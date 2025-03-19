from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os 

#load the environment, this will contain the OPENAI_API_KEY
load_dotenv()

# Initialize the chat model with standard parameters
chat = ChatOpenAI(
    model="gpt-3.5-turbo",    # Model name
    temperature=0.7,          # Adjusts randomness
    max_tokens=100,           # Limits output tokens
    stop=["\n"],              # Stop sequences
)

# Message-based input with roles
conversation = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
]

# Get a response from the chat model
response = chat.invoke(conversation)

# Output the AI's response
print(response.content)
