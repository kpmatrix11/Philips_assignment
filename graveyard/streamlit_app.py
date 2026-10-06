from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graveyard.comparison_assistant import (
    HybridRetriever,
    _load_published_llm,
    answer_question,
    build_document_chunks,
    llm_load_error,
    load_product_records,
)


@st.cache_resource
def build_retriever() -> HybridRetriever:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    records = load_product_records(data_dir)
    documents = build_document_chunks(records)
    return HybridRetriever(documents, product_records=records)


@st.cache_resource
def build_llm():
    return _load_published_llm()


def _render_hit_card(hit: dict, index: int, question: str) -> None:
    brand = hit.get("brand") or "Unknown"
    product_name = hit.get("product_name") or "Product"
    section = hit.get("section") or "Product overview"
    source_url = hit.get("source_url") or ""

    st.markdown(f"### {index}. {product_name}")
    st.caption(f"{brand} • {section}")
    if source_url:
        st.markdown(f"[Open product page]({source_url})")

    snippet = (hit.get("snippet") or "").strip()
    if snippet:
        with st.expander("Matched snippet"):
            st.write(snippet[:800])

    citation = hit.get("citation") or source_url or "No citation available"
    st.caption(citation)


def _render_retrieval_results(hits: list[dict], question: str) -> None:
    if not hits:
        st.info("No public product sources matched this question.")
        return

    grouped = {"Philips Sonicare": [], "Oral-B": []}
    for hit in hits:
        brand = hit.get("brand") or "Unknown"
        if brand in grouped:
            grouped[brand].append(hit)

    st.subheader("Relevant products retrieved")
    for brand in ["Philips Sonicare", "Oral-B"]:
        st.markdown(f"#### {brand}")
        brand_hits = grouped.get(brand, [])
        if not brand_hits:
            st.caption("No relevant match found for this brand.")
            continue

        cols = st.columns(min(2, max(1, len(brand_hits))))
        for i, hit in enumerate(brand_hits[:2]):
            with cols[i % len(cols)]:
                _render_hit_card(hit, i + 1, question)


def main() -> None:
    st.set_page_config(
        page_title="Philips vs Oral-B Comparison",
        page_icon="🪥",
        layout="wide",
    )

    st.title("Oral Care Competitive Intelligence Copilot")
    st.caption("Grounded in the publicly extracted Philips Sonicare and Oral-B product pages in this project dataset.")

    if "question" not in st.session_state:
        st.session_state["question"] = "Compare app connectivity between Philips Sonicare and Oral-B electric toothbrushes."

    retriever = build_retriever()

    with st.sidebar:
        st.subheader("Sample questions")
        sample_questions = [
            "Compare app connectivity between Philips Sonicare and Oral-B electric toothbrushes.",
            "Which brand offers a pressure sensor in its public product pages?",
            "Does Oral-B or Philips claim more plaque removal?",
            "Which brand includes Bluetooth or app-connected features?",
            "Which option is better in the $100-$200 price range?",
        ]
        for sample in sample_questions:
            if st.button(sample, key=f"sample_{sample[:30]}"):
                st.session_state["question"] = sample

    with st.form(key="comparison_form"):
        question = st.text_area(
            "Ask a product question based on the extracted product pages:",
            value=st.session_state["question"],
            height=120,
            placeholder="Example: Compare app connectivity between Philips Sonicare and Oral-B electric toothbrushes.",
        )
        submitted = st.form_submit_button("Ask question")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question before searching.")
            st.stop()

        st.session_state["question"] = question
        with st.spinner("Loading local language model (first load can take a few minutes)..."):
            llm = build_llm()
        with st.spinner("Generating an answer from the local product data..."):
            result = answer_question(
                question,
                retriever,
                llm=llm,
                top_k=5,
            )

        st.markdown("---")
        if "comparison_table" in result:
            st.subheader("Comparison table")
            if result["comparison_table"]:
                st.dataframe(
                    result["comparison_table"],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No matching product metadata was found in the source data.")
            if result.get("llm_used") and result.get("conclusion"):
                st.subheader("LLM conclusion")
                st.write(result["conclusion"])
            else:
                st.subheader("LLM unavailable")
                load_error = llm_load_error()
                detail = f" Model load error: {load_error}" if load_error else " Check that the local model loaded and try again."
                st.warning(f"No LLM-generated conclusion is available.{detail}")
        else:
            st.subheader("Answer")
            st.write(result["answer"])
            _render_retrieval_results(result.get("hits", []), question)
            for warning in result.get("unsupported_claims", []):
                st.warning(warning)

    else:
        st.info("Use the box above to ask a question. The system will retrieve the most relevant Philips and Oral-B products and generate a grounded comparison answer.")


if __name__ == "__main__":
    main()
