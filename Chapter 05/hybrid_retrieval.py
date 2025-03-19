from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
import cassio
from cassio.table.cql import STANDARD_ANALYZER
from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (API keys, etc.) from a .env file
load_dotenv()

# Initialize the cassio connection to Astra DB
cassio.init(
    database_id="YOUR_DATABASE_ID",
    token="YOUR_TOKEN",  # Use your actual token
    keyspace="default_keyspace",  # Replace with your keyspace
)

# Step 1: Create vector store using OpenAI embeddings and Cassandra with hybrid search capabilities
embeddings = OpenAIEmbeddings()
vectorstore = Cassandra(
    embedding=embeddings,
    # New table for storing space exploration data
    table_name="space_exploration",  
    body_index_options=[STANDARD_ANALYZER],
    session=None,
    keyspace=None,
)

# Step 2: Add new, imaginative texts about space exploration
vectorstore.add_texts(
    [
        "In 2025, I visited Mars and explored the Gale Crater.",
        "In 2030, I embarked on a mission to Europa to study its icy oceans.",
        "In 2027, I visited the Moon's South Pole to establish a lunar base.",
        "In 2029, I traveled to the asteroid belt to mine valuable resources.",
        "In 2032, I reached Titan, one of Saturn's moons, to investigate its methane lakes."
    ]
)

# Step 3: Perform a standard similarity search
results = vectorstore.as_retriever().invoke("Where did I explore in 2027?")
print("Results from semantic search")
for result in results:
    print(result.page_content)
print("Results from semantic search ends")
# Output:
# 'In 2027, I visited the Moon's South Pole to establish a lunar base.'

# Step 4: Perform a hybrid search filtering by term "Mars"
results = vectorstore.as_retriever(search_kwargs={"body_search": "Mars"}).invoke(
    "Where did I explore in 2027?"
)
print("Results from keyword search")
for result in results:
    print(result.page_content)
print("Results from keyword search ends")
# Output (filtered to entries mentioning Mars):
# 'In 2025, I visited Mars and explored the Gale Crater.'

# Step 5: Define a custom prompt for the language model (new template)
template = """
You are an expert space explorer, and you have completed several missions to different celestial bodies.
Based on the following context, answer the question as concisely as possible.

Context: {context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Step 6: Initialize the OpenAI language model (GPT-3.5 turbo)
model = ChatOpenAI()

# Step 7: Configure the retriever for hybrid search using `ConfigurableField`
retriever = vectorstore.as_retriever()

configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use for filtering results",
    )
)

# Step 8: Create the RAG chain (combining retrieval and generation)
chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Step 9: Invoke the chain to answer a new question about space exploration
# Query without filtering
response = chain.invoke("What missions did I complete in 2030?")
print(response)
# Expected Output:
# 'In 2030, I embarked on a mission to Europa to study its icy oceans.'

# Query with hybrid search filtering on the term "Moon"
response = chain.invoke(
    "What missions did I complete in 2027?",
    config={"configurable": {"search_kwargs": {"body_search": "Moon"}}},
)
print(response)
