from llama_index.legacy import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore

reader = SimpleDirectoryReader("data/data.csv")
documents = reader.load_data()

print(documents[0])
