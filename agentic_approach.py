import logging
import psycopg2
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, SummaryIndex, \
    get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.retrievers import VectorIndexRetriever, SummaryIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from literalai import LiteralClient

load_dotenv()


logging.basicConfig(
    filename='context_chat_log.log',  # Name of the log file
    filemode='a',  # Append to the file, use 'w' to overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.WARN  # Set the logging level (e.g., DEBUG, INFO)
)

client = LiteralClient()
client.instrument_llamaindex()

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
response_synthesizer = get_response_synthesizer()

vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
# vector_query_engine = vector_index.as_query_engine()
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.3)]
)

vector_tool = QueryEngineTool.from_defaults(
    name="VectorTool",
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from the datasource"
)

summary_index = SummaryIndex.from_documents(documents, storage_context=storage_context)
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
summary_tool = QueryEngineTool.from_defaults(
    name="SummaryTool",
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to datasource"
)


agent_worker = FunctionCallingAgentWorker.from_tools([vector_tool, summary_tool], llm=llm)
agent = AgentRunner(agent_worker)

while True:
    question = input("Ask Question: ").strip()
    if question == "exit":
        break
    response = agent.chat(question)
    print(response)

from llama_index.core import Settings
# 
# print(Settings.text_splitter.class_name())