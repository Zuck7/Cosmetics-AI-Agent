#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for the Cosmetics AI Agent

This provides a lightweight frontend that initializes the RAG system,
the summarization agent and the PlannerAgent, and exposes a simple chat-like
interface for querying the cosmetics knowledge base.

Created to integrate the provided `PlannerAgent` orchestration.
"""

from __future__ import annotations
from typing import Tuple

try:
    import streamlit as st
    _HAVE_STREAMLIT = True
except Exception:
    st = None
    _HAVE_STREAMLIT = False

# Heavy imports are performed lazily inside `init_system` to avoid import-time
# dependency errors in environments where optional packages (PyPDF2, streamlit, etc.)
# are not installed.


def _no_op_decorator(func):
    return func


_cache_decorator = st.cache_resource if _HAVE_STREAMLIT and hasattr(st, "cache_resource") else _no_op_decorator


@_cache_decorator
def init_system(
    data_path: str = "data/cosmetics",
    cosing_csv_path: str = "data/cosing/COSING_Ingredients-Fragrance.Inventory_v2.csv",
) -> Tuple:
    """Initialize documents, vector store, RAG system, summarizer and planner.

    This is cached by Streamlit so initialization is done only once per session.
    """
    # Import components lazily so module import doesn't require heavy deps
    from rag.pdf_loader import load_pdfs
    from rag.cosing_loader import load_cosing
    from rag.vector_store import VectorStore
    from rag.rag_system import RAGSystem
    from summarization_agent.sum_agent import SummarizationAgent
    from planner_agent.planner import PlannerAgent

    # Load documents: PDFs plus the open EU CosIng ingredient database
    docs = load_pdfs(data_path) + load_cosing(cosing_csv_path)

    # Build vector store
    store = VectorStore()
    store.build(docs)

    # Create RAG system and summarizer
    rag = RAGSystem(store)
    summarizer = SummarizationAgent()

    # Create planner agent (loads .env internally)
    planner = PlannerAgent(rag_system=rag, summarizer=summarizer)

    return planner, rag, store, summarizer


class CosmeticsAssistantUI:
    """Manages Streamlit interface and integration with PlannerAgent."""

    def __init__(self):
        # Initialize in run to avoid heavy work at import time
        self.planner = None
        self.rag = None
        self.store = None
        self.summarizer = None

    def run(self):
        if not _HAVE_STREAMLIT:
            raise RuntimeError("Streamlit is required to run the UI. Install it with 'pip install streamlit'.")

        st.set_page_config(page_title="Cosmetics AI Assistant", page_icon="💄")

        st.title("💄 Cosmetics AI Assistant")
        st.markdown("A lightweight frontend for the Cosmetics-AI-Agent planner and RAG system.")

        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            data_path = st.text_input("Knowledge base path", value="data/cosmetics")
            retrieval_k = st.slider("Retrieval chunks (k)", min_value=1, max_value=10, value=4)
            auto_validate = st.checkbox("Auto-validate (if available)", value=False)
            auto_reflect = st.checkbox("Auto-reflect (if available)", value=False)
            max_summary_len = st.slider("Max summary length (words)", min_value=50, max_value=500, value=200)
            show_raw = st.checkbox("Show raw JSON response", value=False)

        # Initialize system (cached)
        try:
            with st.spinner("Initializing system components (may take a few seconds)..."):
                self.planner, self.rag, self.store, self.summarizer = init_system(data_path)
        except Exception as e:
            st.error(f"System initialization failed: {e}")
            return

        # Input area
        st.subheader("Ask a question about cosmetics or ingredients")
        query = st.chat_input("Type your question here and press Enter...")

        if query:
            # show the user message in the chat
            st.chat_message("user")
            st.markdown(query)

            # Run planner
            with st.spinner("Planner is processing your query..."):
                # Planner handles decomposition internally; we can control validation/reflection flags
                try:
                    result = self.planner.coordinate(
                        query,
                        auto_validate=auto_validate,
                        auto_reflect=auto_reflect,
                        retrieval_k=retrieval_k,
                        max_summary_len=max_summary_len,
                    )
                except Exception as e:
                    st.error(f"Error during planning/execution: {e}")
                    return

            # Assistant message
            st.chat_message("assistant")
            # Show formatted output (human-friendly)
            formatted = self.planner.format_output(result)
            st.code(formatted, language="text")

            # Supporting evidence expander
            with st.expander("Supporting evidence / retrieved chunks"):
                chunks = result.get("supporting_evidence", [])
                if not chunks:
                    st.info("No supporting chunks were retrieved.")
                else:
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(c)

            # Metadata
            with st.expander("Metadata"):
                metadata = result.get("metadata", {})
                validation = result.get("validation", {})
                st.json({"metadata": metadata, "validation": validation})

            if show_raw:
                with st.expander("Raw response JSON"):
                    st.json(result)


def main():
    app = CosmeticsAssistantUI()
    app.run()


if __name__ == "__main__":
    main()
