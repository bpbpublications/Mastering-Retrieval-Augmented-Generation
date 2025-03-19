# Import necessary libraries for web scraping, document loading, and embedding generation
# For handling HTML and web scraping
import bs4  
# For creating and using a Chroma vector store
from langchain_chroma import Chroma  
# For loading web content as documents
from langchain_community.document_loaders import WebBaseLoader  
# For parsing string-based outputs from the LLM
from langchain_core.output_parsers import StrOutputParser  
# For handling passthrough inputs in a chain
from langchain_core.runnables import RunnablePassthrough  
# For generating vector embeddings using OpenAI
from langchain_openai import OpenAIEmbeddings  
 # For splitting large texts
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# For using OpenAI’s GPT models
from langchain_openai import OpenAI  
# For creating a prompt template
from langchain_core.prompts import PromptTemplate

# Load environment variables (API keys, etc.) from a .env file
from dotenv import load_dotenv
# This loads the OpenAI API key stored in your .env file
load_dotenv()  


# Step 1: Load the contents of an article from a webpage using WebBaseLoader
# We specify the parts of the webpage we want to parse using BeautifulSoup's 'SoupStrainer'
loader = WebBaseLoader(
    web_paths=("https://www.deepmind.com/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology",),  # New URL for the web content
    
)
docs = loader.load()  # Load the content from the webpage into a document object

# Step 2: Split the loaded document into smaller chunks for better processing
# RecursiveCharacterTextSplitter splits the content into chunks of 1000 characters with some overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)  # Split the document into smaller chunks

# Step 3: Create a vector store using Chroma and the OpenAI embeddings
# Chroma is a vector store that allows efficient similarity search on text embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())  # Store document chunks as vectors

# Step 4: Create a retriever that retrieves relevant chunks of the article based on the query
retriever = vectorstore.as_retriever()

# Step 5: Define a custom prompt for the LLM
# This prompt will be used to guide the language model when generating a response
prompt_template = PromptTemplate(
    input_variables=["question", "context"], 
    template = """
                You are an expert in deep learning, and you have read the article on DeepMind's AlphaFold solution to the protein folding challenge.
                Based on the following context, answer the question comprehensively.

                Context: {context}

                Question: {question}

                Answer:
                """
)

# Helper function to format the retrieved document chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)  # Format the document content as a string

# Step 6: Initialize the OpenAI language model (GPT-3.5 turbo) with specific settings
# 'gpt-3.5-turbo-instruct' is a model designed for generating structured responses
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)  # Set temperature to 0.0 for deterministic answers

# Step 7: Set up the RAG chain (combining retrieval and generation)
# The chain retrieves relevant content, applies the custom prompt, and generates an answer
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Combine retrieval with passthrough question input
    | prompt_template  # Apply the custom prompt
    | llm  # Pass the result to the OpenAI LLM
    | StrOutputParser()  # Parse the output from the LLM into a usable format
)

# Step 8: Query the RAG chain with a new question and print the result
# We ask a question based on the content of the WWF's Living Planet Report 2022
print(rag_chain.invoke("How does AlphaFold utilize deep learning to solve the protein folding problem limit to three sentence?"))
