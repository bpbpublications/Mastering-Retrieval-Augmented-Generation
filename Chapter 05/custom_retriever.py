from typing import List, Dict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class KeywordFrequencyRetriever(BaseRetriever):
    """
    A custom retriever that returns the top k documents based on the frequency of the query keyword in each document.
    
    This retriever supports both synchronous and asynchronous retrieval of documents and prioritizes documents
    where the keyword appears most frequently.
    """
    
    documents: List[Document]
    """ A list of documents to retrieve from."""
    k: int
    """ The maximum number of documents to return."""
    
    def _calculate_keyword_frequency(self, query: str, document: Document) -> int:
        """
        Calculate how many times the query keyword appears in a document.

        Args:
            query: The keyword to search for.
            document: The document to search within.

        Returns:
            The number of times the keyword appears in the document.
        """
        return document.page_content.lower().count(query.lower())

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """ Synchronous method to retrieve the top k documents based on keyword frequency."""
        # Rank documents by keyword frequency
        ranked_docs: List[Dict[str, any]] = [
            {"document": doc, "frequency": self._calculate_keyword_frequency(query, doc)}
            for doc in self.documents
        ]
        # Sort by frequency in descending order and select top k
        ranked_docs.sort(key=lambda x: x["frequency"], reverse=True)
        return [doc["document"] for doc in ranked_docs[:self.k] if doc["frequency"] > 0]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """ Asynchronous version of the retrieval process."""
        # Rank documents by keyword frequency
        ranked_docs: List[Dict[str, any]] = [
            {"document": doc, "frequency": self._calculate_keyword_frequency(query, doc)}
            for doc in self.documents
        ]
        # Sort by frequency in descending order and select top k
        ranked_docs.sort(key=lambda x: x["frequency"], reverse=True)
        return [doc["document"] for doc in ranked_docs[:self.k] if doc["frequency"] > 0]


# Example usage:
if __name__ == "__main__":
    # Sample documents
    docs = [
        Document(page_content="Apple is a technology company known for its iPhones."),
        Document(page_content="The apple fruit is nutritious and contains vitamin C."),
        Document(page_content="Bananas are rich in potassium and offer quick energy."),
        Document(page_content="Apple has revolutionized the smartphone industry."),
    ]
    
    # Create a retriever that fetches the top 2 documents
    retriever = KeywordFrequencyRetriever(documents=docs, k=2)
    
    # Test the synchronous retrieval
    query = "apple"
    relevant_docs = retriever._get_relevant_documents(query=query, run_manager=None)
    for doc in relevant_docs:
        print(f"Sync Retrieved: {doc.page_content}")
    
    # Test the asynchronous retrieval
    import asyncio
    async def async_test():
        async_relevant_docs = await retriever._aget_relevant_documents(query=query, run_manager=None)
        for doc in async_relevant_docs:
            print(f"Async Retrieved: {doc.page_content}")

    asyncio.run(async_test())
