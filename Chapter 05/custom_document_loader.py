import sqlite3
from typing import Iterator, AsyncIterator
from datetime import datetime
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CustomDocumentLoader(BaseLoader):
    """A document loader that retrieves data with metadata from an in-memory SQLite database."""

    def __init__(self) -> None:
        """Initialize the loader by creating an in-memory SQLite database."""
        # Establish connection to an in-memory SQLite database
        self.conn = sqlite3.connect(":memory:")
        self._create_schema_and_insert_data()

    def _create_schema_and_insert_data(self):
        """Creates tables and inserts data with metadata."""
        cursor = self.conn.cursor()

        # Create tables: one for documents and one for metadata
        cursor.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                author_id INTEGER,
                creation_date TIMESTAMP,
                type TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)

        # Insert authors
        cursor.execute("INSERT INTO authors (name) VALUES (?)", ('Alice',))
        cursor.execute("INSERT INTO authors (name) VALUES (?)", ('Bob',))

        # Insert documents with metadata using parameterized queries
        cursor.execute("""
            INSERT INTO documents (content, author_id, creation_date, type)
            VALUES (?, ?, ?, ?)
        """, ("This is Alice's first document.", 1, datetime.now(), 'report'))

        cursor.execute("""
            INSERT INTO documents (content, author_id, creation_date, type)
            VALUES (?, ?, ?, ?)
        """, ("Bob wrote this second document.", 2, datetime.now(), 'memo'))

        cursor.execute("""
            INSERT INTO documents (content, author_id, creation_date, type)
            VALUES (?, ?, ?, ?)
        """, ("This is another report by Alice.", 1, datetime.now(), 'report'))

        self.conn.commit()

    def lazy_load(self) -> Iterator[Document]:
        """Lazily loads documents with metadata from the in-memory SQLite database."""
        cursor = self.conn.cursor()

        # Join documents with author information and retrieve metadata
        cursor.execute("""
            SELECT d.id, d.content, a.name, d.creation_date, d.type
            FROM documents d
            JOIN authors a ON d.author_id = a.id
        """)
        for row in cursor.fetchall():
            doc_id, content, author_name, creation_date, doc_type = row
            yield Document(
                page_content=content,
                metadata={
                    "document_id": doc_id,
                    "author": author_name,
                    "creation_date": creation_date,
                    "type": doc_type,
                    "source": "in_memory_db"
                }
            )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Asynchronously loads documents with metadata."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT d.id, d.content, a.name, d.creation_date, d.type
            FROM documents d
            JOIN authors a ON d.author_id = a.id
        """)
        for row in cursor.fetchall():
            doc_id, content, author_name, creation_date, doc_type = row
            yield Document(
                page_content=content,
                metadata={
                    "document_id": doc_id,
                    "author": author_name,
                    "creation_date": creation_date,
                    "type": doc_type,
                    "source": "in_memory_db"
                }
            )

# Create an instance of the in-memory loader
loader = CustomDocumentLoader()

# Test the lazy load interface
for doc in loader.lazy_load():
    print(doc)
