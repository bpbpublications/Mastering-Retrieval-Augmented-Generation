from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables (such as API keys) from a .env file
from dotenv import load_dotenv
load_dotenv()

# Define the model and prompt template
model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("What is the best way to reach {city} from LA?")

# Create a chain to fetch weather for San Francisco
chain = prompt | model | StrOutputParser()

analysis_prompt = ChatPromptTemplate.from_template("How convenient is the suggested route? Suggested route: {route}")

composed_chain = {"route": chain} | analysis_prompt | model | StrOutputParser()

print(composed_chain.invoke({"city": "San Francisco"}))
