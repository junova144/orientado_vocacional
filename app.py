# app.py

import streamlit as st
from rag_agent import build_agent
from langchain_core.messages import HumanMessage

# Título principal
st.title("🧭 Orientador Vocacional basado en Inteligencia Artificial 🤖")

# Subtítulo
st.write("Inspirado en el libro 'El Elemento' - Ken Robinson")

# Texto guía encima de la caja de input
st.write("Mientras más preguntes, más claro se vuelve el sendero de tu camino. Vamos, ¿qué te impide descubrir tu pasión?.")

query = st.text_input("Pregunta:")

if st.button("Consultar"):
    if not query:
        st.warning("Por favor ingresa una pregunta.")
    else:
        agent = build_agent()

        with st.spinner("Buscando información..."):
            config = {"configurable": {"thread_id": "user"}}
            response = None

            for step in agent.stream(
                {"messages": [HumanMessage(content=query)]},
                config,
                stream_mode="values",
            ):
                response = step["messages"][-1].content

        st.subheader("Respuesta del agente:")
        st.write(response)



