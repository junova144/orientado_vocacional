# rag_agent.py

import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore as LE

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


@tool
def orientador_vocacional(query: str) -> str:
    """Consulta información sobre orientación vocacional en Elasticsearch."""
    vector_store = LE(
        es_url=os.environ["ES_URL"],
        es_user=os.environ["ES_USER"],
        es_password=os.environ["ES_PASSWORD"],
        index_name="rag_ov",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=OpenAIEmbedding()
    )

    retriever = index.as_retriever(search_kwargs={"k": 4})
    results = retriever.retrieve(query)

    output = "\n\n".join([
        f"[{i+1}] {node.node.text}\nFuente: {node.node.metadata.get('source', 'N/A')}"
        for i, node in enumerate(results)
    ])
    return output


def build_agent():
    model = ChatOpenAI(model="gpt-4.1")

    system_prompt = """Eres un asistente de orientación vocacional.
Usa solo tu herramienta RAG para responder. Si no tienes información, dilo."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{messages}")
    ])

    tools = [orientador_vocacional]

    agent = create_react_agent(model, tools, prompt=prompt)
    return agent
