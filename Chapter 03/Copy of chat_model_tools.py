from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def weather_func(location:str) -> str:
    """Get the current weather information"""
    return f"The weather in {location} is sunny and 25°C."


# Initialize the chat model with standard parameters
chat = ChatOpenAI(
    model="gpt-3.5-turbo-1106",    # Model name
)

llm_with_tools = chat.bind_tools([weather_func])

conversation = [
    SystemMessage(content="You are a chatbot that can interact with tools. Use my tool weather_func"),
    HumanMessage(content="What's the weather like in SanFrancisco?"),
]

# Pass the conversation to the chat model, expecting the tool to be used
ai_msg = llm_with_tools.invoke(conversation)

messages = []

# Output the response that includes tool interaction
for tool_call in ai_msg.tool_calls:
    selected_tool = {"weather_func": weather_func}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

print(messages)
