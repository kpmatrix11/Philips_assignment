import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "comparison_assistant.py"

spec = importlib.util.spec_from_file_location("comparison_assistant", MODULE_PATH)
comparison_assistant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison_assistant)


def test_load_product_records_reads_both_brands():
    records = comparison_assistant.load_product_records(ROOT / "data")
    brands = {record["brand"] for record in records}
    assert {"Philips Sonicare", "Oral-B"}.issubset(brands)
    assert len(records) >= 20


def test_chunk_documents_include_source_metadata():
    records = comparison_assistant.load_product_records(ROOT / "data")
    documents = comparison_assistant.build_document_chunks(records)
    assert documents
    sample = documents[0]
    assert "source_url" in sample
    assert "brand" in sample
    assert "product_name" in sample
    assert "citation" in sample
    assert "http" in sample["citation"].lower()

    feature_doc = next((doc for doc in documents if "App connectivity" in doc.get("section", "")), None)
    assert feature_doc is not None
    assert "http" in feature_doc["citation"].lower()
    assert "app connectivity" in feature_doc["citation"].lower()


def test_hybrid_search_returns_relevant_results():
    records = comparison_assistant.load_product_records(ROOT / "data")
    documents = comparison_assistant.build_document_chunks(records)
    retriever = comparison_assistant.HybridRetriever(documents)
    hits = retriever.search("app connectivity", top_k=3)
    assert hits
    assert any(hit["brand"] in {"Philips Sonicare", "Oral-B"} for hit in hits)


def test_safe_answer_refuses_unknowns_and_preserves_citations():
    records = comparison_assistant.load_product_records(ROOT / "data")
    documents = comparison_assistant.build_document_chunks(records)
    retriever = comparison_assistant.HybridRetriever(documents)
    answer = comparison_assistant.safe_answer_question(
        "Which model has 1000% plaque removal and app connectivity?",
        retriever,
        llm=None,
    )
    assert "citation" in answer.lower() or "source" in answer.lower()
    assert "not found in sources" in answer.lower() or "I could not" in answer.lower() or "Based on the sources" in answer.lower()


def test_embedding_text_and_metadata_are_separated():
    records = comparison_assistant.load_product_records(ROOT / "data")
    documents = comparison_assistant.build_document_chunks(records)
    assert documents
    assert all("embedding_text" in doc for doc in documents)
    assert all(isinstance(doc["metadata"], dict) for doc in documents)
    assert documents[0]["metadata"]["brand"] == documents[0]["brand"]

    retriever = comparison_assistant.HybridRetriever(documents)
    hits = retriever.search("app connectivity", top_k=1)
    assert hits
    assert "metadata" in hits[0]
    assert hits[0]["metadata"]["product_name"] == hits[0]["product_name"]


def test_best_by_rating_and_price_range_are_supported():
    records = comparison_assistant.load_product_records(ROOT / "data")
    philips_best = comparison_assistant.select_best_product(records, brand="Philips Sonicare")
    assert philips_best is not None
    assert "detail_title" in philips_best
    assert philips_best["rating_value"] >= 4.0

    budget_matches = comparison_assistant.filter_products_by_price(records, 0, 149)
    assert budget_matches
    assert all(0 <= item["price_usd"] <= 149 for item in budget_matches)
    assert any(item["detail_title"] for item in budget_matches)


def test_comparison_returns_all_products_and_grounds_llm_in_table():
    records = [
        {
            "brand": "Philips Sonicare",
            "product_name": "Philips Model A",
            "brush_modes_count": "3",
            "product_url": "https://usa.philips.com/a",
        },
        {
            "brand": "Philips Sonicare",
            "product_name": "Philips Model B",
            "brush_modes_count": "4",
            "product_url": "https://usa.philips.com/b",
        },
        {
            "brand": "Oral-B",
            "product_name": "Oral-B Model C",
            "brush_mode_count": "5",
            "product_url": "https://oralb.com/c",
        },
    ]
    retriever = object.__new__(comparison_assistant.HybridRetriever)
    retriever.product_records = records
    captured_prompt = {}

    def fake_llm(prompt):
        captured_prompt["text"] = prompt
        return [{"generated_text": "Direct answer: The listed models have three, four, and five modes respectively."}]

    result = comparison_assistant.answer_question(
        "Compare brush modes between Philips Sonicare and Oral-B.",
        retriever,
        llm=fake_llm,
    )

    assert len(result["comparison_table"]) == 3
    assert [row["Brush modes"] for row in result["comparison_table"]] == [3, 4, 5]
    assert "all matching products; values are source metadata" in captured_prompt["text"]
    assert "Philips Model B" in captured_prompt["text"]
    assert result["conclusion"].startswith("The listed models")


def test_comparison_prompt_applies_method_to_exact_query():
    records = [
        {
            "brand": "Philips Sonicare",
            "product_name": "Philips Plaque Model",
            "plaque_claims": "Up to 10x more plaque removal than a manual toothbrush.",
            "product_url": "https://usa.philips.com/plaque-model",
        },
        {
            "brand": "Oral-B",
            "product_name": "Oral-B Plaque Model",
            "plaque_claims": "100% more plaque removal than a manual brush.",
            "product_url": "https://oralb.com/plaque-model",
        },
    ]
    retriever = object.__new__(comparison_assistant.HybridRetriever)
    retriever.product_records = records
    captured_prompt = {}

    def fake_llm(prompt):
        captured_prompt["text"] = prompt
        return [{"generated_text": "Direct answer: Philips states up to 10x more plaque removal; Oral-B states 100% more."}]

    question = "Does Oral-B or Philips claim more plaque removal?"
    result = comparison_assistant.answer_question(question, retriever, llm=fake_llm)

    assert question in captured_prompt["text"]
    assert "Apply the same method to any feature named in the user query" in captured_prompt["text"]
    assert "Example of the method, not an assumption about the current query" in captured_prompt["text"]
    assert "identify each brand's highest supported value or claim" in captured_prompt["text"]
    assert "Philips Plaque Model" in captured_prompt["text"]
    assert "Oral-B Plaque Model" in captured_prompt["text"]
    assert result["llm_used"] is True
    assert result["conclusion"].startswith("Philips states")


def test_missing_or_failed_llm_does_not_return_static_conclusion():
    retriever = object.__new__(comparison_assistant.HybridRetriever)
    retriever.product_records = [
        {
            "brand": "Philips Sonicare",
            "product_name": "Philips Model",
            "brush_modes_count": 3,
            "product_url": "https://usa.philips.com/model",
        }
    ]

    def failing_llm(_prompt):
        raise RuntimeError("Model generation failed")

    for llm in (None, failing_llm):
        result = comparison_assistant.answer_question(
            "Compare brush modes.",
            retriever,
            llm=llm,
        )
        assert result["conclusion"] is None
        assert result["answer"] is None
        assert result["llm_used"] is False


def test_llm_loader_exposes_load_error_and_clears_it_after_success(monkeypatch):
    def failing_pipeline(*_args, **_kwargs):
        raise RuntimeError("test model load failure")

    monkeypatch.setattr(comparison_assistant, "pipeline", failing_pipeline)
    assert comparison_assistant._load_published_llm() is None
    assert comparison_assistant.llm_load_error() == "RuntimeError: test model load failure"

    monkeypatch.setattr(comparison_assistant, "pipeline", lambda *_args, **_kwargs: lambda _prompt, **_options: [])
    assert comparison_assistant._load_published_llm() is not None
    assert comparison_assistant.llm_load_error() is None


def test_loaded_llm_uses_short_generation_limit(monkeypatch):
    generation_options = {}

    def fake_pipeline(*_args, **_kwargs):
        def fake_generator(_prompt, **options):
            generation_options.update(options)
            return [{"generated_text": "Grounded answer."}]

        return fake_generator

    monkeypatch.setattr(comparison_assistant, "pipeline", fake_pipeline)
    llm = comparison_assistant._load_published_llm()

    assert llm is not None
    assert llm("Question") == [{"generated_text": "Grounded answer."}]
    assert generation_options["max_new_tokens"] == 96
    assert generation_options["return_full_text"] is False


def test_combined_brand_top_rated_query_returns_direct_top_product():
    records = comparison_assistant.load_product_records(ROOT / "data")
    documents = comparison_assistant.build_document_chunks(records)
    retriever = comparison_assistant.HybridRetriever(documents)
    answer = comparison_assistant.safe_answer_question(
        "From both brands, fetch me the top rated product",
        retriever,
        top_k=6,
    )
    assert answer
    assert any(keyword in answer.lower() for keyword in ["top rated", "highest-rated", "highest rated", "rating"])
    assert any(brand in answer for brand in ["Philips Sonicare", "Oral-B"])


def test_hit_answer_is_only_llm_output_and_uses_matched_evidence():
    question = "Does this product support app connectivity?"
    hit = {
        "brand": "Philips Sonicare",
        "product_name": "Test model",
        "snippet": "App connectivity: Bluetooth connects this toothbrush to the mobile app.",
    }
    expected_answer = "Yes. Bluetooth connects this toothbrush to the mobile app."
    captured_prompt = {}

    def fake_llm(prompt):
        captured_prompt["text"] = prompt
        return [{"generated_text": f"Answer: {expected_answer}"}]

    answer = comparison_assistant.generate_hit_relevance_summary(question, hit, llm=fake_llm)

    assert answer == expected_answer
    assert question in captured_prompt["text"]
    assert hit["snippet"] in captured_prompt["text"]
    assert "relevance explanation" in captured_prompt["text"]


def test_hit_answer_is_unavailable_without_llm_output():
    hit = {"snippet": "App connectivity: Bluetooth connectivity."}

    assert comparison_assistant.generate_hit_relevance_summary("Is it connected?", hit) is None
    assert comparison_assistant.generate_hit_relevance_summary(
        "Is it connected?",
        hit,
        llm=lambda _prompt: [],
    ) is None
