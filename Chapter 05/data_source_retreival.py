# Import necessary libraries
# For loading text files
from langchain_community.document_loaders import TextLoader  
# For creating vector-based document stores
from langchain_community.vectorstores import FAISS 
# For using OpenAI embeddings          
from langchain_openai import OpenAIEmbeddings 
# For splitting long documents into smaller chunks               
from langchain_text_splitters import CharacterTextSplitter   

# Load environment variables (such as API keys) from a .env file
from dotenv import load_dotenv
load_dotenv()  # This ensures the API keys (e.g., OpenAI keys) are loaded

# Load a sample text document (replace "sample_article.txt" with your actual text file)
loader = TextLoader("sample_article.txt")

# Load the document content
documents = loader.load()

# Split the document into chunks of 1000 characters without any overlap
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Initialize OpenAI embeddings to convert the text into vectors
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from the chunks of text and their embeddings
vectorstore = FAISS.from_documents(texts, embeddings)

# Convert the FAISS vector store into a retriever object
retriever = vectorstore.as_retriever()

# Ask a question to retrieve relevant chunks of text
docs = retriever.invoke("What did the article say about the solar power initiative?")

# Output the retrieved documents
print(docs)
