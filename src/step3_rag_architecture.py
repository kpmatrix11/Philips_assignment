from dotenv import load_dotenv

import os
import json
import re
from typing import Optional, Literal

from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# ============================================================
# Environment
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN is not set in .env"
    )


# ============================================================
# Configuration
# ============================================================

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "toothbrush_products"


# ============================================================
# Embeddings
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Chroma
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)


# ============================================================
# LLM
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=600,
    temperature=0.0,
)

chat = ChatHuggingFace(llm=llm)


# ============================================================
# Query Plan Schema
# ============================================================

class QueryPlan(BaseModel):

    intent: Literal[
        "lookup",
        "comparison",
        "ranking",
        "filter",
        "aggregation",
        "general",
    ]

    operation: Optional[
        Literal[
            "max",
            "min",
            "greater_than",
            "less_than",
            "equals",
            "contains",
            "between",
        ]
    ] = None

    attribute: Optional[str] = None

    group_by: Optional[str] = None

    brands: list[str] = Field(
        default_factory=list
    )

    products: list[str] = Field(
        default_factory=list
    )

    requested_attributes: list[str] = Field(
        default_factory=list
    )

    filters: list[dict] = Field(
        default_factory=list
    )


# ============================================================
# Relevance Judge Schema
# ============================================================

class RelevanceVerdict(BaseModel):

    relevance: Literal[
        "Relevant",
        "Not relevant",
    ]

    reason: str


# ============================================================
# Query Planner Prompt
# ============================================================

QUERY_PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a query planning component for a product RAG system.

The product dataset contains Philips Sonicare and Oral-B
electric toothbrush products.

Your job is ONLY to understand the user's question and
return a JSON query plan.

Do NOT answer the user's question.

Available attributes are:

- price_usd
- product_category
- customer_rating
- customer_rating_count
- app_connectivity
- plaque_claims
- gum_claims
- brush_modes_count
- short_description
- long_description

Possible intents:

- lookup
- comparison
- ranking
- filter
- aggregation
- general

Possible operations:

- max
- min
- greater_than
- less_than
- equals
- contains
- between

Rules:

1. Use "ranking" when the user asks for highest, lowest,
   top, best-rated, cheapest, most expensive, most reviewed,
   most modes, etc.

2. Use "filter" when the user specifies a condition such as:
   below $100, above 4 stars, more than 1000 reviews, etc.

3. Use "comparison" when the user wants to compare products,
   brands, or attributes.

4. Use "lookup" when the user asks for information about
   a particular product.

5. Use "general" when no structured operation is appropriate.

6. For "top rated", use:
      operation = "max"
      attribute = "customer_rating"

7. For "cheapest", use:
      operation = "min"
      attribute = "price_usd"

8. For "most expensive", use:
      operation = "max"
      attribute = "price_usd"

9. For "most reviewed", use:
      operation = "max"
      attribute = "customer_rating_count"

10. For "most brush modes", use:
      operation = "max"
      attribute = "brush_modes_count"

11. If the user says "for both brands", use:
      group_by = "brand"

12. Only put attributes explicitly requested or required
    to answer the question into requested_attributes.

13. Do not invent product names.

14. Do not answer the question.

15. NUMERIC RANGE RULE:

    If the user specifies BOTH a lower and upper bound,
    always use:

      intent = "filter"
      operation = "between"

    Put BOTH conditions into the filters array.

    The lower bound must use:
      operator = ">="

    The upper bound must use:
      operator = "<="

    Examples:

    "between $99 and $199"
    "from 99 dollars to 199 dollars"
    "ranging from 99 to 199"
    "$99-$199"
    "products priced 99 to 199 dollars"

    All of these should produce:

    "operation": "between"

    with:

    "filters": [
      {{"operator": ">=", "value": 99}},
      {{"operator": "<=", "value": 199}}
    ]

16. For a single numeric condition, use the appropriate
    operation and ONE filter.

    Examples:

    "below $100"
    ->
    operation = "less_than"
    ->
    filters = [{{"operator": "<", "value": 100}}]

    "under $100"
    ->
    operation = "less_than"
    ->
    filters = [{{"operator": "<", "value": 100}}]

    "above $100"
    ->
    operation = "greater_than"
    ->
    filters = [{{"operator": ">", "value": 100}}]

    "at least $100"
    ->
    operation = "greater_than"
    ->
    filters = [{{"operator": ">=", "value": 100}}]

    "at most $100"
    ->
    operation = "less_than"
    ->
    filters = [{{"operator": "<=", "value": 100}}]

17. Do not convert a range into a single greater_than
    or less_than condition.

18. For numeric conditions, use numeric values in filters,
    without currency symbols or text.

19. If the user specifies brands, identify them in the
    "brands" field.

20. If the user specifies particular products, identify them
    in the "products" field.

Return ONLY valid JSON.

Example 1:

User:
top rated product for both brands

JSON:

{{
  "intent": "ranking",
  "operation": "max",
  "attribute": "customer_rating",
  "group_by": "brand",
  "brands": ["Philips Sonicare", "Oral-B"],
  "products": [],
  "requested_attributes": ["customer_rating"],
  "filters": []
}}

Example 2:

User:
products ranging from 99 dollars to 199 dollars

JSON:

{{
  "intent": "filter",
  "operation": "between",
  "attribute": "price_usd",
  "group_by": null,
  "brands": [],
  "products": [],
  "requested_attributes": ["price_usd"],
  "filters": [
    {{"operator": ">=", "value": 99}},
    {{"operator": "<=", "value": 199}}
  ]
}}

Example 3:

User:
toothbrushes below 100 dollars

JSON:

{{
  "intent": "filter",
  "operation": "less_than",
  "attribute": "price_usd",
  "group_by": null,
  "brands": [],
  "products": [],
  "requested_attributes": ["price_usd"],
  "filters": [
    {{"operator": "<", "value": 100}}
  ]
}}

Example 4:

User:
Oral-B toothbrushes between $100 and $200

JSON:

{{
  "intent": "filter",
  "operation": "between",
  "attribute": "price_usd",
  "group_by": null,
  "brands": ["Oral-B"],
  "products": [],
  "requested_attributes": ["price_usd"],
  "filters": [
    {{"operator": ">=", "value": 100}},
    {{"operator": "<=", "value": 200}}
  ]
}}

User question:

{question}
"""
)


# ============================================================
# Normalize Values
# ============================================================

def normalize_brand(brand):

    if not brand:
        return None

    value = brand.lower().strip()

    value = value.replace("-", " ")

    if "philips" in value or "sonicare" in value:
        return "Philips Sonicare"

    if "oral b" in value or "oral-b" in value:
        return "Oral-B"

    return brand


def normalize_attribute(attribute):

    if not attribute:
        return None

    value = attribute.lower().strip()

    aliases = {

        "price": "price_usd",
        "cost": "price_usd",
        "price_usd": "price_usd",

        "rating": "customer_rating",
        "ratings": "customer_rating",
        "customer rating": "customer_rating",
        "customer_rating": "customer_rating",

        "reviews": "customer_rating_count",
        "review count": "customer_rating_count",
        "number of reviews": "customer_rating_count",
        "customer_rating_count": "customer_rating_count",

        "app": "app_connectivity",
        "app connectivity": "app_connectivity",
        "app_connectivity": "app_connectivity",

        "plaque": "plaque_claims",
        "plaque claims": "plaque_claims",
        "plaque_claims": "plaque_claims",

        "gum": "gum_claims",
        "gum claims": "gum_claims",
        "gum_claims": "gum_claims",

        "brush modes": "brush_modes_count",
        "modes": "brush_modes_count",
        "brush_modes_count": "brush_modes_count",

        "category": "product_category",
        "product category": "product_category",
        "product_category": "product_category",

        "description": "short_description",
        "short description": "short_description",
        "short_description": "short_description",

        "long description": "long_description",
        "long_description": "long_description",
    }

    return aliases.get(value, value)


# ============================================================
# Claim Query Detection
#
# These queries should use semantic retrieval because
# plaque/gum claims are textual product claims rather than
# deterministic numeric product attributes.
# ============================================================

def is_claim_query(question):

    if not question:
        return False

    text = question.lower().strip()

    claim_terms = [
        "plaque",
        "gum",
        "gingiv",
        "plaque claim",
        "gum claim",
        "plaque removal",
        "gum protection",
    ]

    return any(
        term in text
        for term in claim_terms
    )


# ============================================================
# Parse JSON
# ============================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```\s*",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in model response."
        )

    return text[start:end + 1]


# ============================================================
# Parse Query Plan
# ============================================================

def plan_query(question):

    messages = QUERY_PLANNER_PROMPT.invoke({
        "question": question
    })

    try:

        response = chat.invoke(messages)

        raw = response.content

        json_text = extract_json(raw)

        data = json.loads(json_text)

        plan = QueryPlan.model_validate(data)

    except Exception:

        # Safe fallback to normal hybrid RAG
        plan = QueryPlan(
            intent="general",
            requested_attributes=[],
        )

    # Normalize attribute
    if plan.attribute:

        plan.attribute = normalize_attribute(
            plan.attribute
        )

    # Normalize requested attributes
    plan.requested_attributes = [
        normalize_attribute(attribute)
        for attribute in plan.requested_attributes
        if normalize_attribute(attribute)
    ]

    # Normalize brands
    plan.brands = [
        normalize_brand(brand)
        for brand in plan.brands
        if normalize_brand(brand)
    ]

    # Normalize group
    if plan.group_by:

        plan.group_by = (
            plan.group_by
            .lower()
            .strip()
        )

    return plan


# ============================================================
# Load All Documents from Chroma
# ============================================================

all_docs = vectorstore.get()


# ============================================================
# Build BM25 Index
# ============================================================

bm25_documents = []

for i, text in enumerate(
    all_docs["documents"]
):

    bm25_documents.append({

        "text": text,

        "metadata": all_docs["metadatas"][i],

    })


bm25_corpus = [
    doc["text"].lower().split()
    for doc in bm25_documents
]

bm25 = BM25Okapi(
    bm25_corpus
)


# ============================================================
# Safe Comparison
# ============================================================

def compare_values(
    product_value,
    condition_value,
    operator,
):

    """
    Compare existing values without converting or parsing them.

    If the values cannot be safely compared for the requested
    operation, raise an error so retrieval can fall back to
    hybrid search.
    """

    if product_value is None or condition_value is None:
        raise ValueError("Missing value for comparison.")

    try:

        if operator == "<":
            return product_value < condition_value

        if operator == "<=":
            return product_value <= condition_value

        if operator == ">":
            return product_value > condition_value

        if operator == ">=":
            return product_value >= condition_value

        if operator in ("=", "=="):
            return product_value == condition_value

        if operator == "contains":

            if not isinstance(product_value, str) or not isinstance(condition_value, str):
                raise TypeError("Contains comparison requires string values.")

            return condition_value.strip().lower() in product_value.strip().lower()

        raise ValueError(
            f"Unsupported comparison operator: {operator}"
        )

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "Could not safely compare the existing values "
            f"{product_value!r} and {condition_value!r}."
        ) from exc


# ============================================================
# Build Product-Level Catalog
# ============================================================

def build_product_catalog(all_docs):

    products = {}

    for metadata in all_docs["metadatas"]:

        product_name = metadata.get(
            "product_name"
        )

        brand = metadata.get(
            "brand"
        )

        if not product_name or not brand:
            continue

        if product_name not in products:

            products[product_name] = {

                "product_name": product_name,

                "brand": normalize_brand(
                    brand
                ),

                "price_usd": None,

                "customer_rating": None,

                "customer_rating_count": None,

                "app_connectivity": None,

                "plaque_claims": None,

                "gum_claims": None,

                "brush_modes_count": None,

                "product_category": None,

            }

        attribute = metadata.get(
            "attribute"
        )

        value = metadata.get(
            "source_snippet"
        )

        if value is None:
            continue

        # ----------------------------------------------------
        # Preserve the existing source value exactly as provided.
        # No numeric parsing, conversion, or reformatting is done.
        # ----------------------------------------------------

        if attribute in products[product_name]:

            products[product_name][
                attribute
            ] = value

    return products


product_catalog = build_product_catalog(
    all_docs
)


# ============================================================
# Structured Retrieval
# ============================================================

def structured_retrieval(plan):

    candidates = list(
        product_catalog.values()
    )

    # ========================================================
    # Brand Filtering
    # ========================================================

    if plan.brands:

        requested_brands = {
            normalize_brand(brand)
            for brand in plan.brands
        }

        candidates = [

            product

            for product in candidates

            if normalize_brand(
                product["brand"]
            ) in requested_brands

        ]

    # ========================================================
    # Product Filtering
    # ========================================================

    if plan.products:

        requested_products = {

            product.lower().strip()

            for product in plan.products

        }

        candidates = [

            product

            for product in candidates

            if product["product_name"]
            .lower()
            .strip()
            in requested_products

        ]

    # ========================================================
    # Ranking
    # ========================================================

    if plan.intent == "ranking":

        attribute = normalize_attribute(
            plan.attribute
        )

        if not attribute:
            return []

        candidates = [

            product

            for product in candidates

            if product.get(attribute) is not None

        ]

        if not candidates:
            return []

        # ----------------------------------------------------
        # Ranking grouped by brand
        # ----------------------------------------------------

        if plan.group_by == "brand":

            selected = {}

            for product in candidates:

                brand = normalize_brand(
                    product["brand"]
                )

                current = selected.get(
                    brand
                )

                if current is None:

                    selected[brand] = product

                else:

                    current_value = current.get(
                        attribute
                    )

                    product_value = product.get(
                        attribute
                    )

                    if current_value is None or product_value is None:
                        raise ValueError(
                            f"Missing value for ranking attribute '{attribute}'."
                        )

                    try:

                        if (
                            plan.operation == "max"
                            and product_value > current_value
                        ):

                            selected[brand] = product

                        elif (
                            plan.operation == "min"
                            and product_value < current_value
                        ):

                            selected[brand] = product

                    except TypeError as exc:

                        raise ValueError(
                            "Could not safely compare existing values "
                            f"for ranking attribute '{attribute}'."
                        ) from exc

            candidates = list(
                selected.values()
            )

        # ----------------------------------------------------
        # Ranking globally
        # ----------------------------------------------------

        else:

            if not all(
                isinstance(product.get(attribute), (int, float))
                and not isinstance(product.get(attribute), bool)
                for product in candidates
            ):

                raise ValueError(
                    "Structured ranking cannot safely compare existing "
                    f"values for attribute '{attribute}'."
                )

            if plan.operation == "max":

                candidates = [
                    max(
                        candidates,
                        key=lambda product: product[attribute],
                    )
                ]

            elif plan.operation == "min":

                candidates = [
                    min(
                        candidates,
                        key=lambda product: product[attribute],
                    )
                ]

            else:

                raise ValueError(
                    f"Unsupported ranking operation: {plan.operation}"
                )

    # ========================================================
    # Filters
    # ========================================================

    if plan.intent == "filter":

        attribute = normalize_attribute(
            plan.attribute
        )

        if not attribute:
            return []

        if not plan.filters:

            return []

        # ----------------------------------------------------
        # Apply every filter condition.
        #
        # This creates AND logic for ranges.
        # ----------------------------------------------------

        for condition in plan.filters:

            operator = condition.get(
                "operator"
            )

            value = condition.get(
                "value"
            )

            if value is None:
                continue

            filtered = []

            for product in candidates:

                product_value = product.get(
                    attribute
                )

                if product_value is None:
                    continue

                passed = compare_values(
                    product_value,
                    value,
                    operator,
                )

                if passed:

                    filtered.append(
                        product
                    )

            candidates = filtered

            if not candidates:

                break

    # ========================================================
    # Retrieve Evidence ONLY for Selected Products
    # ========================================================

    selected_docs = []

    requested_attributes = [

        normalize_attribute(attribute)

        for attribute in (
            plan.requested_attributes or []
        )

        if normalize_attribute(attribute)

    ]

    for product in candidates:

        product_name = product[
            "product_name"
        ]

        if requested_attributes:

            search_query = " ".join(
                requested_attributes
            )

        else:

            search_query = product_name

        try:

            docs = vectorstore.similarity_search(

                search_query,

                k=25,

                filter={
                    "product_name": product_name
                },

            )

        except Exception:

            docs = vectorstore.similarity_search(

                f"{product_name} {search_query}",

                k=25,

            )

            docs = [

                doc

                for doc in docs

                if doc.metadata.get(
                    "product_name"
                ) == product_name

            ]

        # ----------------------------------------------------
        # Keep only requested attributes
        # ----------------------------------------------------

        if requested_attributes:

            attribute_docs = []

            for doc in docs:

                doc_attribute = doc.metadata.get(
                    "attribute"
                )

                normalized_doc_attribute = (
                    normalize_attribute(
                        doc_attribute
                    )
                )

                if (
                    normalized_doc_attribute
                    in requested_attributes
                ):

                    attribute_docs.append(
                        doc
                    )

            if attribute_docs:

                docs = attribute_docs

            else:

                docs = []

        selected_docs.extend(
            docs
        )

    return selected_docs


# ============================================================
# Hybrid Retrieval
# ============================================================

def hybrid_retrieval(
    question,
    k=10,
    vector_k=10,
    keyword_k=10,
):

    # ========================================================
    # Vector retrieval
    # ========================================================

    vector_docs = vectorstore.similarity_search(
        question,
        k=vector_k,
    )

    # ========================================================
    # BM25 retrieval
    # ========================================================

    query_tokens = (
        question.lower().split()
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    keyword_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:keyword_k]

    keyword_docs = [
        bm25_documents[i]
        for i in keyword_indices
    ]

    # ========================================================
    # Reciprocal Rank Fusion
    # ========================================================

    scores = {}

    documents = {}

    # --------------------------------------------------------
    # Vector ranking
    # --------------------------------------------------------

    for rank, doc in enumerate(
        vector_docs
    ):

        doc_id = (
            doc.metadata.get(
                "document_id"
            )
            or doc.metadata.get(
                "source_id"
            )
            or doc.page_content
        )

        scores[doc_id] = (
            scores.get(
                doc_id,
                0
            )
            + (
                1
                / (
                    60
                    + rank
                    + 1
                )
            )
        )

        documents[doc_id] = doc

    # --------------------------------------------------------
    # Keyword ranking
    # --------------------------------------------------------

    for rank, item in enumerate(
        keyword_docs
    ):

        doc_id = (
            item["metadata"].get(
                "document_id"
            )
            or item["metadata"].get(
                "source_id"
            )
            or item["text"]
        )

        scores[doc_id] = (
            scores.get(
                doc_id,
                0
            )
            + (
                1
                / (
                    60
                    + rank
                    + 1
                )
            )
        )

        if doc_id not in documents:

            documents[doc_id] = Document(
                page_content=item["text"],
                metadata=item["metadata"],
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids[:k]
    ]


# ============================================================
# Main Retrieval Dispatcher
# ============================================================

def retrieve(
    question,
    plan,
    k=10,
    vector_k=10,
    keyword_k=10,
):

    # ========================================================
    # Claim-based queries
    #
    # Plaque/gum claims are textual claims and should be
    # handled through semantic + keyword retrieval rather
    # than deterministic numeric filtering.
    # ========================================================

    if is_claim_query(question):

        return hybrid_retrieval(
            question,
            k=k,
            vector_k=vector_k,
            keyword_k=keyword_k,
        )

    # ========================================================
    # Structured intents
    # ========================================================

    structured_intents = {
        "ranking",
        "filter",
        "aggregation",
    }

    if plan.intent in structured_intents:

        try:

            docs = structured_retrieval(
                plan
            )

        except Exception:

            # Any structured-retrieval error falls back to
            # hybrid retrieval, which then follows the normal
            # generation path in ask().
            docs = []

        if docs:

            return docs

        # ----------------------------------------------------
        # Structured retrieval produced no usable evidence.
        # Fall back to hybrid retrieval.
        # ----------------------------------------------------

        return hybrid_retrieval(
            question,
            k=k,
            vector_k=vector_k,
            keyword_k=keyword_k,
        )

    # ========================================================
    # Default semantic retrieval
    # ========================================================

    return hybrid_retrieval(
        question,
        k=k,
        vector_k=vector_k,
        keyword_k=keyword_k,
    )


# ============================================================
# Build Context
# ============================================================

def build_context(docs):

    context_parts = []

    sources = {}

    for doc in docs:

        metadata = doc.metadata

        source_id = metadata.get(
            "source_id"
        )

        document_id = metadata.get(
            "document_id"
        )

        product_name = metadata.get(
            "product_name"
        )

        brand = metadata.get(
            "brand"
        )

        attribute = metadata.get(
            "attribute"
        )

        source_url = metadata.get(
            "source_url"
        )

        source_snippet = metadata.get(
            "source_snippet"
        )

        # ----------------------------------------------------
        # Save source information internally
        # ----------------------------------------------------

        if source_id not in sources:

            sources[source_id] = {

                "product_name":
                    product_name,

                "brand":
                    brand,

                "attributes":
                    [],

                "source_url":
                    source_url,

                "source_snippets":
                    [],

            }

        if (
            attribute
            not in sources[source_id]["attributes"]
        ):

            sources[source_id][
                "attributes"
            ].append(
                attribute
            )

        if (
            source_snippet
            not in sources[source_id][
                "source_snippets"
            ]
        ):

            sources[source_id][
                "source_snippets"
            ].append(
                source_snippet
            )

        # ----------------------------------------------------
        # Context supplied to LLM
        # ----------------------------------------------------

        context_parts.append(

            f"""
SOURCE_ID:
{source_id}

DOCUMENT_ID:
{document_id}

Brand:
{brand}

Product:
{product_name}

Attribute:
{attribute}

SOURCE_SNIPPET:
{source_snippet}

SOURCE_URL:
{source_url}

SOURCE_EVIDENCE:
{doc.page_content}
""".strip()

        )

    return (
        "\n\n".join(context_parts),
        sources,
    )


# ============================================================
# Answer Prompt
# ============================================================

SYSTEM_PROMPT = ChatPromptTemplate.from_template(
    """
You are a product comparison assistant.

Your job is to answer questions about Philips Sonicare
and Oral-B electric toothbrushes using ONLY the supplied
retrieved evidence.

============================================================
STRICT GROUNDING RULES
============================================================

1. Use ONLY the evidence supplied below.

2. Do NOT use your own knowledge.

3. Do NOT guess.

4. Do NOT infer missing product features.

5. Do NOT treat missing information as "No".

6. If requested information is not present in the
   supplied evidence, use:

   "Not found in the provided US sources."

7. Never invent a SOURCE_ID, DOCUMENT_ID, SOURCE_SNIPPET,
   SOURCE_URL, product name, feature, or value.

8. Every product-specific factual claim must be directly
   supported by the supplied evidence.

9. Use the exact SOURCE_SNIPPET from the evidence as the
   Citation for the corresponding row.

10. The Citation must NOT be paraphrased.

11. The Citation must be the specific snippet that directly
    supports the Product + Feature + Value in that row.

12. If multiple snippets support the same row, select the
    single most directly relevant SOURCE_SNIPPET.

13. Do not combine snippets from unrelated products.

14. Do not use a citation from one product to support another
    product.

15. Do not provide medical or dental advice.

16. You may report manufacturer/product claims when they
    are present in the supplied evidence.

17. Do not turn manufacturer claims into medical advice.

18. Do not state that one product is medically superior.

============================================================
QUERY PLAN
============================================================

The application generated the following query plan:

{query_plan}

The query plan determines what information the user requested.

Do NOT add unrelated attributes.

For example:

- If the user asks for the top-rated product, do NOT
  automatically include price.

- If the user asks about app connectivity, do NOT
  automatically include price or rating.

Only include an attribute when:

1. The user explicitly requested it, OR

2. It is necessary to answer the question.

For ranking questions, use the products selected by the
application.

Do NOT independently choose another product.

All attributes shown for a product must belong to that
same exact product.

============================================================
SOURCE EVIDENCE
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER REQUIREMENTS
============================================================

Return the answer ONLY as a Markdown table.

The table MUST have exactly these columns:

| Product Name | Brand | Feature | Value | Citation | Product Link |

Rules for the table:

1. Product Name:
   Use the exact product name from the supplied evidence.

2. Brand:
   Use the exact brand from the supplied evidence.

3. Feature:
   Use the requested feature/attribute.

4. Value:
   Give only the value supported by the evidence.

5. Citation:
   Use the EXACT SOURCE_SNIPPET from the evidence that
   supports that particular Product + Feature + Value.

   Do NOT paraphrase the snippet.

   Do NOT use SOURCE_ID as the citation.

   Do NOT create a citation yourself.

6. Product Link:
   Use the exact SOURCE_URL associated with that product.

   Format it as:

   [Product page](SOURCE_URL)

7. If multiple products answer the question, create one
   row for each product.

8. If multiple features are explicitly requested for a
    product, create a row for each product-feature pair.

9. Do NOT mix information between products.

10. Do NOT create a separate Sources section.

11. Do NOT create a USED_SOURCES section.

12. Do NOT output SOURCE_ID separately.

13. Do NOT output DOCUMENT_ID separately.

14. Do NOT output URLs outside the Product Link column.

15. Do NOT add explanatory text before or after the table.

16. Do NOT add any columns other than the eight specified
    columns.

17. If evidence is insufficient for a product/feature,
    write:

    Not found in the provided US sources.

18. The relevance reasoning must NOT introduce information
    that is absent from the evidence.

Example format:

| Product Name | Brand | Feature | Value | Citation | Product Link |
|---|---|---|---|---|---|---|---|
| Example Product | Philips Sonicare | Battery operation | Yes | Exact source snippet here | [Product page](https://example.com) |

Return ONLY the table.
"""
)


# ============================================================
# Independent Relevance Judge Prompt
# ============================================================

RELEVANCE_JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """
You are an independent relevance judge for a product RAG assistant.

Judge whether the generated answer directly and completely
addresses the user's question and whether its product-specific
claims are supported by the retrieved evidence.

Use ONLY:

1. The user question
2. The retrieved evidence
3. The generated answer

Do NOT use pretrained knowledge.

Do NOT use outside product knowledge.

Do NOT use web search.

============================================================
RELEVANCE CRITERIA
============================================================

Mark "Relevant" only when:

1. The answer addresses the user's actual question.

2. The requested product(s), brand(s), feature(s), filter,
   comparison, ranking, or other operation are addressed.

3. Product-specific claims are supported by the retrieved
   evidence.

4. Each Citation corresponds to the product and feature
   shown in that row.

5. Each Product Link corresponds to the same product shown
   in that row.

6. The answer does not mix information between products.

7. The answer does not introduce unsupported claims.

8. The Relevance Reasoning is consistent with the supplied
   evidence.

Mark "Not relevant" when:

1. The answer is off-topic.

2. A material part of the question is unanswered.

3. The wrong product or brand is provided.

4. A requested filter/ranking/comparison was not correctly
   addressed.

5. A product-specific claim is unsupported.

6. Citation evidence does not support the corresponding row.

7. Product Link belongs to a different product.

8. Information from different products is incorrectly mixed.

9. The answer relies on information outside the supplied
   evidence.

A response that correctly says information is missing should
not be marked Not relevant merely because the requested
information was unavailable.

Minor wording differences alone should not cause
"Not relevant".

Return ONLY valid JSON with exactly these keys:

{{
  "relevance": "Relevant",
  "reason": "Short explanation."
}}

USER QUESTION:

{question}

RETRIEVED EVIDENCE:

{context}

GENERATED ANSWER:

{answer}
"""
)


# ============================================================
# Relevance Judge
# ============================================================

def judge_relevance(
    question,
    context,
    answer,
):

    messages = RELEVANCE_JUDGE_PROMPT.invoke({

        "question":
            question,

        "context":
            context,

        "answer":
            answer,

    })

    try:

        response = chat.invoke(
            messages
        )

        verdict_data = json.loads(
            extract_json(
                response.content
            )
        )

        verdict = (
            RelevanceVerdict
            .model_validate(
                verdict_data
            )
        )

        return verdict.model_dump()

    except Exception as exc:

        return {

            "relevance":
                "Not relevant",

            "reason":
                (
                    "Judge could not validate the answer; "
                    "defaulted to Not relevant. "
                    f"{type(exc).__name__}: {exc}"
                ),

        }


# ============================================================
# Generate Answer
# ============================================================

def ask(question):

    # --------------------------------------------------------
    # 1. Understand query
    # --------------------------------------------------------

    plan = plan_query(
        question
    )

    # --------------------------------------------------------
    # 2. Retrieve
    # --------------------------------------------------------

    docs = retrieve(

        question,

        plan,

        k=25,

        vector_k=25,

        keyword_k=25,

    )

    # --------------------------------------------------------
    # 3. Nothing found
    # --------------------------------------------------------

    if not docs:

        return {

            "answer":
                (
                    "| Product Name | Brand | Feature | Value | "
                    "Citation | Relevance | Relevance Reasoning | "
                    "Product Link |\n"
                    "|---|---|---|---|---|---|---|---|\n"
                    "| Not found | Not found | Not found | "
                    "Not found in the provided US sources. | "
                    "Not found | Not relevant | "
                    "No supporting product evidence was retrieved. | "
                    "Not found |"
                ),

            "sources":
                {},

            "relevance":
                "Not relevant",

            "relevance_reason":
                "No supporting product evidence was retrieved.",

            "query_plan":
                plan.model_dump(),

        }

    # --------------------------------------------------------
    # 4. Build context
    # --------------------------------------------------------

    context, sources = build_context(
        docs
    )

    # --------------------------------------------------------
    # 5. Build answer prompt
    # --------------------------------------------------------

    messages = SYSTEM_PROMPT.invoke({

        "context":
            context,

        "question":
            question,

        "query_plan":
            json.dumps(
                plan.model_dump(),
                indent=2,
            ),

    })

    # --------------------------------------------------------
    # 6. Generate answer
    # --------------------------------------------------------

    response = chat.invoke(
        messages
    )

    answer = response.content.strip()

    # --------------------------------------------------------
    # 7. Independent relevance judgment
    # --------------------------------------------------------

    relevance_verdict = judge_relevance(

        question,

        context,

        answer,

    )

    # --------------------------------------------------------
    # 8. Return result
    # --------------------------------------------------------

    return {

        "answer":
            answer,

        "sources":
            sources,

        "relevance":
            relevance_verdict[
                "relevance"
            ],

        "relevance_reason":
            relevance_verdict[
                "reason"
            ],

        "query_plan":
            plan.model_dump(),

    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "Philips Sonicare vs Oral-B RAG Assistant"
    )

    print(
        "=========================================="
    )

    print(
        "\nType 'exit' to quit."
    )

    while True:

        question = input(
            "\nUser Query: "
        )

        if question.lower() == "exit":
            break

        try:

            result = ask(
                question
            )

            print(
                "\nAssistant:\n"
            )

            print(
                result["answer"]
            )

            print(
                f"\nRelevance: "
                f"{result['relevance']}"
            )

            print(
                "Judge reason:",
                result[
                    "relevance_reason"
                ],
            )

        except Exception as exc:

            print(
                "\nError:",
                exc
            )