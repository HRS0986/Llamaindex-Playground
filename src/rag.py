import psycopg2
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from dotenv import load_dotenv
from rag_config import RagConfig

from sqlalchemy import make_url

load_dotenv()

class RAG:
    def __init__(self):
        self.config = RagConfig()
        self.connection = self.__initialize_pgvector_connection()
        self.vs_config = self.config.vector_store
        self.vector_store: PGVectorStore = self.__initialize_vector_store()
        self.query_engine = self.create_vector_index()

    def __initialize_pgvector_connection(self):
        connection_string = self.vs_config["connection_string"]
        db_name = self.vs_config["db_name"]
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"CREATE DATABASE {db_name} IF NOT EXISTS")
        return conn

    def __initialize_vector_store(self) -> PGVectorStore:
        url = make_url(self.vs_config["connection_string"])
        vector_store = PGVectorStore.from_params(
            database=self.vs_config["db_name"],
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=self.vs_config["table_name"],
            embed_dim=1536,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )
        return vector_store

    def create_vector_index(self, documents: list[Document]):
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        query_engine = index.as_query_engine()
        return query_engine

    def load_documents(self, file_path: str):
        reader = SimpleDirectoryReader(file_path)
        documents = reader.load_data()
        return documents


if __name__ == '__main__':
    success, connection = initialize_pgvector()
    print(success, connection)
