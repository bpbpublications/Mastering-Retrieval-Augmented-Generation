# Import necessary modules from LangChain for OpenAI model, agents, and tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

# Import search tool and other utilities from LangChain's community tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# Import utilities to define a custom prompt template
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# Load environment variables (such as API keys) from a .env file
from dotenv import load_dotenv
load_dotenv()

# Define the search tool using Tavily API to search online with a max of 2 results
search = TavilySearchResults(max_results=2)

# Load documents from a webpage and split them into smaller chunks for indexing
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

# Create a vector index from the document chunks using FAISS and OpenAI embeddings
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
# Turn the vector index into a retriever tool to handle document queries
retriever = vector.as_retriever()

# Create a retriever tool for LangSmith documentation, allowing the agent to search it
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",  # Tool name
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# Define a custom prompt template to guide the agent's behavior during interaction
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),  # System message defining the agent's role
        MessagesPlaceholder(variable_name='chat_history', optional=True),  # Memory for chat history
        ("human", "{input}"),  # Input message placeholder for user queries
        MessagesPlaceholder(variable_name='agent_scratchpad')  # Placeholder for internal agent processing
    ]
)

# Initialize the language model (GPT-4 in this case) for the agent
model = ChatOpenAI(model="gpt-4")

# Combine the tools (search tool and retriever tool) into a list for the agent
tools = [search, retriever_tool]  # Custom tools defined earlier

# Create the agent with the model, tools, and prompt to determine actions based on user input
agent = create_tool_calling_agent(model, tools, prompt)

# AgentExecutor coordinates the agent's actions and tool execution
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Invoke the agent to process a query about the weather in San Francisco and print the result
response = agent_executor.invoke({"input": "What's the weather in SF?"})
print(response["output"])  # Output the agent's response to the console
