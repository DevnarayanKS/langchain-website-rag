"""Streamlit URL RAG Chat application.

Run with:
    streamlit run app.py

The application expects OPENAI_API_KEY to be set in the environment.
"""

import os
import uuid
from urllib.parse import urlparse

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def is_valid_url(url):
    """Check that the user supplied an HTTP/HTTPS URL."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


@st.cache_resource
def load_llm():
    """Initialise the LLM and embeddings once per app process."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model=MODEL_NAME, max_tokens=500)
    return embeddings, llm


def build_knowledge_base(url):
    """Load a URL, split its content, embed it, and create a retriever."""
    loader = WebBaseLoader(url)
    docs = loader.load()

    if not docs:
        raise ValueError("No content could be loaded from this URL.")

    # Split the webpage into smaller chunks for better retrieval.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        raise ValueError("The URL was loaded, but no text chunks were created.")

    embeddings, _ = load_llm()

    # Create an in-memory Chroma collection for this URL.
    # This avoids mixing documents from different URLs.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    return retriever, len(docs), len(chunks)


def format_docs(docs):
    """Combine retrieved document chunks into prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question, chat_history, retriever, llm):
    """Rewrite follow-ups for retrieval, then answer from retrieved context."""

    # Step 1: Convert a follow-up question into a standalone search question.
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

    standalone_question = (
        rewrite_prompt | llm | StrOutputParser()
    ).invoke(
        {
            "chat_history": chat_history[-6:],
            "question": question,
        }
    )

    # Step 2: Retrieve relevant chunks from the URL.
    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    # Step 3: Answer using only the retrieved URL content.
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

    answer = (
        answer_prompt | llm | StrOutputParser()
    ).invoke(
        {
            "context": context,
            "chat_history": chat_history[-6:],
            "question": question,
        }
    )

    return answer, docs


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="URL RAG Chat", page_icon="🔗")

st.title("🔗 URL RAG Chat")
st.caption("Enter a webpage URL, let the app build its knowledge base, and then ask questions about it.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set in your environment.")
    st.stop()


# Initialise session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "source_url" not in st.session_state:
    st.session_state.source_url = None

if "ready" not in st.session_state:
    st.session_state.ready = False


# Sidebar controls.
with st.sidebar:
    st.header("Knowledge Base")

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com/article",
        help="Enter the webpage you want the RAG system to learn from.",
    )

    process_url = st.button(
        "Process URL",
        type="primary",
        use_container_width=True,
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.ready and st.session_state.source_url:
        st.success("Knowledge base ready")
        st.caption(f"Source: {st.session_state.source_url}")


# Process a new URL only when the user clicks the button.
if process_url:
    if not is_valid_url(url):
        st.error("Please enter a valid URL starting with http:// or https://")
    else:
        # Clear the previous conversation because it belongs to another source.
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.retriever = None
        st.session_state.llm = None
        st.session_state.ready = False
        st.session_state.source_url = None

        with st.status("Building knowledge base...", expanded=True) as status:
            try:
                st.write("Loading webpage...")
                retriever, document_count, chunk_count = build_knowledge_base(url)

                st.write("Loading language model...")
                _, llm = load_llm()

                st.session_state.retriever = retriever
                st.session_state.llm = llm
                st.session_state.source_url = url
                st.session_state.ready = True

                status.update(
                    label="Knowledge base ready!",
                    state="complete",
                    expanded=False,
                )

                st.success(
                    f"Processed the URL successfully: "
                    f"{document_count} document(s) → {chunk_count} chunks."
                )

            except Exception as error:
                status.update(
                    label="Failed to process URL",
                    state="error",
                    expanded=True,
                )
                st.error(f"Unable to process this URL: {error}")


# Display previous chat messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Do not allow Q/A until a URL has been processed.
if not st.session_state.ready:
    st.info("Enter a URL in the sidebar and click **Process URL** to start asking questions.")
else:
    if question := st.chat_input("Ask a question about the webpage"):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the knowledge base..."):
                    answer, docs = answer_question(
                        question,
                        st.session_state.chat_history,
                        st.session_state.retriever,
                        st.session_state.llm,
                    )

                st.markdown(answer)

                with st.expander("Sources"):
                    for index, doc in enumerate(docs, start=1):
                        source = doc.metadata.get(
                            "source",
                            st.session_state.source_url,
                        )
                        st.markdown(f"{index}. {source}")

            except Exception as error:
                st.error(f"Unable to answer the question: {error}")
                st.stop()

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.session_state.chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )
