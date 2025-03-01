from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge import AgentKnowledge
from agno.vectordb.pgvector import PgVector

def get_mlx_knowledge_base(db_url: str) -> AgentKnowledge:
    """Initialize knowledge base for MLX and Wan2.1 documentation"""
    return AgentKnowledge(
        vector_db=PgVector(
            db_url=db_url,
            table_name="mlx_t2v_docs",
            schema="ai",
            embedder=OpenAIEmbedder(id="text-embedding-ada-002", dimensions=1536),
        ),
        num_documents=5,  # Retrieve 5 most relevant documents
    ) 