"""Streamlit chat UI for the Kochi RAG project.

Run with: streamlit run app.py
The application expects OPENAI_API_KEY to be set in the environment.
"""

import os
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_resources():
    """Open the existing persisted Chroma collection and initialise the LLM."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
    )
    llm = ChatOpenAI(model=MODEL_NAME, max_tokens=500)
    return retriever, llm


def format_docs(docs):
    """Combine Chroma document chunks into prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question, chat_history, retriever, llm):
    """Rewrite follow-ups for retrieval, then answer from retrieved context."""
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the latest user question as a standalone search question "
                "using the conversation only when needed. Return only the question.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    standalone_question = (rewrite_prompt | llm | StrOutputParser()).invoke(
        {"chat_history": chat_history[-6:], "question": question}
    )

    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using only the supplied context. "
                "If the context does not contain the answer, say that you do not "
                "have enough information in the provided context.\n\n"
                "Context:\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    answer = (answer_prompt | llm | StrOutputParser()).invoke(
        {
            "context": context,
            "chat_history": chat_history[-6:],
            "question": question,
        }
    )
    return answer, docs


st.set_page_config(page_title="Kochi RAG Chat", page_icon="💬")
st.title("💬 Kochi RAG Chat")
st.caption("Ask questions about the Kochi knowledge base.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set in your environment.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about Kochi"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            retriever, llm = load_resources()
            with st.spinner("Searching the knowledge base..."):
                answer, docs = answer_question(
                    question,
                    st.session_state.chat_history,
                    retriever,
                    llm,
                )
            st.markdown(answer)

            with st.expander("Sources"):
                for index, doc in enumerate(docs, start=1):
                    st.markdown(f"{index}. {doc.metadata.get('source', 'Unknown source')}")
        except Exception as error:
            st.error(f"Unable to answer the question: {error}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend(
        [HumanMessage(content=question), AIMessage(content=answer)]
    )
