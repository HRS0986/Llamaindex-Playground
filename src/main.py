import logging
import psycopg2
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.base.llms.types import ChatMessage
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from sqlalchemy import make_url
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

load_dotenv()

logging.basicConfig(
    filename='context_chat_log.log',  # Name of the log file
    filemode='a',  # Append to the file, use 'w' to overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.WARN  # Set the logging level (e.g., DEBUG, INFO)
)

connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
db_name = "vector_store_db"
table_name = "windows_events"
conn = psycopg2.connect(connection_string)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
conn.autocommit = True

with conn.cursor() as cur:
    # Check if the database exists
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone()

    if not exists:
        # Create the database
        cur.execute(f'CREATE DATABASE {db_name}')
        print(f"Database '{db_name}' created successfully.")
    else:
        print(f"Database '{db_name}' already exists.")

url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=table_name,
    embed_dim=1536,
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)
llm = OpenAI(model="gpt-4o-mini")
documents = SimpleDirectoryReader("./data").load_data()

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
synthesizer = get_response_synthesizer()

retriever_query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
prefix_messages = [ChatMessage.from_str("You are a chatbot. You can answer any question, only based on the provided context.")]

context_chat_engine = ContextChatEngine(
    retriever=retriever,
    memory=memory,
    llm=llm,
    prefix_messages=prefix_messages,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

chat_engine = index.as_chat_engine()

while True:
    question = input("Ask Question: ").strip()
    logging.warning(f"Question: {question}")
    if question == "exit":
        break
    response = context_chat_engine.chat(question)
    # response = retriever_query_engine.query(question)
    # response = chat_engine.chat(question)
    logging.warning(f"Response: {response}")
    print(f"Response: {response}")
