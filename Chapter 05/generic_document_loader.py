import sqlite3
from typing import Iterator
from langchain_core.document_loaders import BaseBlobParser, Blob, BaseLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_core.documents import Document


class SQLiteBlobLoader(BaseLoader):
    """A custom blob loader that reads binary data (blobs) from an SQLite in-memory database."""

    def __init__(self) -> None:
        # Create in-memory SQLite connection
        self.conn = sqlite3.connect(":memory:")
        self._create_schema_and_insert_blob_data()

    def _create_schema_and_insert_blob_data(self):
        """Create a table and insert binary data into the in-memory SQLite database."""
        cursor = self.conn.cursor()

        # Create table for binary data
        cursor.execute("""
            CREATE TABLE binary_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content BLOB NOT NULL
            )
        """)

        # Insert some sample binary data (blobs)
        cursor.execute("INSERT INTO binary_data (content) VALUES (?)", (b'This is a binary blob.',))
        cursor.execute("INSERT INTO binary_data (content) VALUES (?)", (b'More binary data here.',))
        self.conn.commit()

    def yield_blobs(self) -> Iterator[Blob]:
        """Yields blobs from the SQLite database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content FROM binary_data")
        
        for row in cursor.fetchall():
            blob_id, content = row
            yield Blob(data=content, metadata={"blob_id": blob_id, "source": "in_memory_db"})

# Custom blob parser for SQLite blobs
class SQLiteBlobParser(BaseBlobParser):
    """Parses blobs from SQLite and converts them into documents."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse the blob data into document objects."""
        yield Document(
            page_content=blob.data.decode('utf-8'),
            metadata={"blob_id": blob.metadata["blob_id"], "source": blob.metadata["source"]}
        )


# Initialize the SQLite blob loader
blob_loader = SQLiteBlobLoader()

# Initialize the SQLite blob parser
parser = SQLiteBlobParser()

# Use GenericLoader to load blobs from the SQLite database and parse them into documents
loader = GenericLoader(blob_loader=blob_loader, blob_parser=parser)

# Test the loader by lazily loading documents
for doc in loader.lazy_load():
    print(doc)
