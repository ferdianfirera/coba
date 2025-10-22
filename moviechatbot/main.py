# moviemaster.py
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# ---- secrets (works both locally via .env and on Streamlit Cloud via st.secrets) ----
if "QDRANT_URL" in st.secrets:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set. For local runs put it in a .env file; on Streamlit put it in the app secrets.")
    st.stop()

collection_name = "movie_collection"

# ---- LLM + Embeddings ----
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.4)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# ---- connect to existing Qdrant collection ----
try:
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
except Exception as e:
    st.error(
        "Could not connect to Qdrant collection. Make sure you ran ingestion and that QDRANT_URL/QDRANT_API_KEY are correct.\nError: %s" % e
    )
    st.stop()

retriever = qdrant.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.6})

# Custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are MovieMaster, a friendly movie expert.\n"
        "If asked about a movie, describe it.\n"
        "If asked for recommendations, list 3–5 titles and gives its rating.\n"
        "Be concise and engaging.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "Answer in English:"
    )
)

# ---- Memory (this makes the chatbot remember conversation!) ----
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)


# ---- Streamlit UI ----
st.set_page_config(page_title="MovieMaster — IMDB RAG Chat", layout="wide")
st.title("🎬 Cinemabot — AI Movie Companion")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, bot) tuples

with st.sidebar:
    st.header("Instructions")
    st.write(
        "Hello, im an movie assistan\n Ask for movie recommendations or the details of the movie.\n\n"
        "If you want to clear the conversation, click the button below."
    )
    if st.button("Clear chat"):
        st.session_state.history = []

# input area
prompt = st.chat_input("Ask me a movie question")

# Display past messages
for role, content in st.session_state.history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

if prompt:
    # append user message to history and UI
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching..."):
        # get the response from the RAG chain
        chat_pairs = []
        last_user = None
        for role, text in st.session_state.history:
            if role == "user":
                last_user = text
            elif role == "assistant" and last_user:
                chat_pairs.append((last_user, text))
                last_user = None

        # Run RAG chain
        result = qa_chain({"question": prompt, "chat_history": chat_pairs})
        answer = result.get("answer", "")
        st.session_state.history.append(("assistant", answer))

        # Display assistant answer
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Display retrieved sources
        sources = result.get("source_documents", [])
        if sources:
            st.markdown("**Sources:**")
            for doc in sources:
                meta = doc.metadata or {}
                title = meta.get("title", "Unknown Title")
                snippet = (doc.page_content or "").replace("", " ")[:300]
        st.write(f"- **{title}** — {snippet}")