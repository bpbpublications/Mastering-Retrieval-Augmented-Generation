from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv  # Load environment variables
from langgraph.checkpoint.memory import MemorySaver  # In-memory checkpointer

# Load environment variables (like API keys) from a .env file
load_dotenv()

# Define a custom tool to perform a simple operation
@tool
def custom_function(input: int) -> int:
    """Performs a custom operation on the input by adding 2 to it."""
    return input + 2

# Define the system message that will guide the agent's behavior
system_message = "You are a helpful assistant. Please respond politely."

# Initialize the OpenAI model (using GPT-4)
model = ChatOpenAI(model="gpt-4")

# List of tools available to the agent
tools = [custom_function]

# Create a memory saver to enable state persistence across interactions
memory = MemorySaver()

# Create a LangGraph agent with the model, tools, system message, and memory
agent_with_memory = create_react_agent(
    model, tools, state_modifier=system_message, checkpointer=memory
)

# Define configuration, including thread ID for session tracking
config = {"configurable": {"thread_id": "session-123"}}

# Query 1: Ask the agent to perform a custom function operation
response1 = agent_with_memory.invoke(
    {
        "messages": [
            ("user", "Hello! Can you tell me the result of custom_function(7)?")
        ]
    },
    config
)["messages"][-1].content
print(f"Response 1: {response1}")
print("---")

# Query 2: Ask the agent if it remembers the user's name
response2 = agent_with_memory.invoke(
    {"messages": [("user", "Can you remember my name?")]},
    config
)["messages"][-1].content
print(f"Response 2: {response2}")
print("---")

# Query 3: Ask the agent to recall the previous output
response3 = agent_with_memory.invoke(
    {"messages": [("user", "What was the result of the custom function earlier?")]},
    config
)["messages"][-1].content
print(f"Response 3: {response3}")
