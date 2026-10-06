import json
import re

from step3_rag_architecture import (
    ask,
    plan_query,
    retrieve,
)


# ============================================================
# 15 Evaluation Questions
# ============================================================

TEST_CASES = [
    {
        "id": "Q01",
        "question": "Which Philips Sonicare product has the highest customer rating?",
        "expected_attribute": "customer_rating",
        "expected_brand": "Philips Sonicare",
    },
    {
        "id": "Q02",
        "question": "Which Oral-B product has the highest customer rating?",
        "expected_attribute": "customer_rating",
        "expected_brand": "Oral-B",
    },
    {
        "id": "Q03",
        "question": "Which Philips Sonicare product is the cheapest?",
        "expected_attribute": "price_usd",
        "expected_brand": "Philips Sonicare",
    },
    {
        "id": "Q04",
        "question": "Which Oral-B product is the cheapest?",
        "expected_attribute": "price_usd",
        "expected_brand": "Oral-B",
    },
    {
        "id": "Q05",
        "question": "Which product has the most brush modes?",
        "expected_attribute": "brush_modes_count",
    },
    {
        "id": "Q06",
        "question": "Compare Philips Sonicare and Oral-B on price and customer rating.",
        "expected_attribute": "price_usd",
    },
    {
        "id": "Q07",
        "question": "Compare Philips Sonicare and Oral-B on brush modes.",
        "expected_attribute": "brush_modes_count",
    },
    {
        "id": "Q08",
        "question": "Compare Philips Sonicare and Oral-B on app connectivity.",
        "expected_attribute": "app_connectivity",
    },
    {
        "id": "Q09",
        "question": "Which products cost more than $100?",
        "expected_attribute": "price_usd",
    },
    {
        "id": "Q10",
        "question": "Which products have a customer rating of 4.5 or higher?",
        "expected_attribute": "customer_rating",
    },
    {
        "id": "Q11",
        "question": "Which products have more than 1000 customer reviews?",
        "expected_attribute": "customer_rating_count",
    },
    {
        "id": "Q12",
        "question": "What are the plaque-related claims for Philips Sonicare products?",
        "expected_attribute": "plaque_claims",
        "expected_brand": "Philips Sonicare",
    },
    {
        "id": "Q13",
        "question": "What are the gum-related claims for Oral-B products?",
        "expected_attribute": "gum_claims",
        "expected_brand": "Oral-B",
    },
    {
        "id": "Q14",
        "question": "Which products support app connectivity?",
        "expected_attribute": "app_connectivity",
    },
    {
        "id": "Q15",
        "question": "Compare Philips Sonicare and Oral-B products using price, rating, and brush modes.",
        "expected_attribute": "price_usd",
    },
]


# ============================================================
# Expected Output Columns
# ============================================================

EXPECTED_COLUMNS = [
    "Product Name",
    "Brand",
    "Feature",
    "Value",
    "Citation",
    "Relevance",
    "Relevance Reasoning",
    "Product Link",
]


# ============================================================
# Parse Markdown Table
# ============================================================

def parse_table(answer):
    rows = []

    if not answer:
        return rows

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip().startswith("|")
    ]

    if len(lines) < 2:
        return rows

    for line in lines[2:]:
        parts = [
            x.strip()
            for x in line.strip("|").split("|")
        ]

        if len(parts) != 8:
            continue

        rows.append(
            dict(
                zip(
                    EXPECTED_COLUMNS,
                    parts,
                )
            )
        )

    return rows


# ============================================================
# Check 1: Output Format
# ============================================================

def check_output_format(answer):

    if not answer:
        return False

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip().startswith("|")
    ]

    if len(lines) < 2:
        return False

    header = [
        x.strip()
        for x in lines[0].strip("|").split("|")
    ]

    return header == EXPECTED_COLUMNS


# ============================================================
# Check 2: Citations Present
# ============================================================

def check_citations_present(rows):

    if not rows:
        return False

    return all(
        row["Citation"].strip()
        for row in rows
    )


# ============================================================
# Check 3: Product URLs Present
# ============================================================

def check_urls_present(rows):

    if not rows:
        return False

    return all(
        (
            "http://" in row["Product Link"]
            or "https://" in row["Product Link"]
        )
        for row in rows
    )


# ============================================================
# Check 4: Relevance Values
# ============================================================

def check_relevance_values(rows):

    if not rows:
        return False

    allowed = {
        "Relevant",
        "Not relevant",
    }

    return all(
        row["Relevance"] in allowed
        for row in rows
    )


# ============================================================
# Check 5: Retrieval Hit
#
# IMPORTANT:
# retrieve() requires:
#
# retrieve(question, plan, ...)
#
# ============================================================

def check_retrieval_hit(
    test_case,
    retrieved_docs,
):

    expected_attribute = test_case.get(
        "expected_attribute"
    )

    expected_brand = test_case.get(
        "expected_brand"
    )

    for doc in retrieved_docs:

        metadata = doc.metadata

        attribute = str(
            metadata.get(
                "attribute",
                "",
            )
        ).lower()

        brand = str(
            metadata.get(
                "brand",
                "",
            )
        ).lower()

        attribute_match = (
            expected_attribute is None
            or attribute == expected_attribute.lower()
        )

        brand_match = (
            expected_brand is None
            or expected_brand.lower() in brand
            or brand in expected_brand.lower()
        )

        if attribute_match and brand_match:
            return True

    return False


# ============================================================
# Check 6: Citation Grounding
# ============================================================

def check_citation_grounding(
    rows,
    retrieved_docs,
):

    snippets = set()

    for doc in retrieved_docs:

        snippet = doc.metadata.get(
            "source_snippet"
        )

        if snippet:
            snippets.add(
                str(snippet).strip()
            )

    checked = 0
    grounded = 0

    for row in rows:

        citation = row["Citation"].strip()

        if not citation:
            continue

        checked += 1

        if citation in snippets:
            grounded += 1

    rate = (
        grounded / checked
        if checked
        else 0.0
    )

    return {
        "checked": checked,
        "grounded": grounded,
        "rate": rate,
    }


# ============================================================
# Check 7: Product Grounding
# ============================================================

def check_product_grounding(
    rows,
    retrieved_docs,
):

    retrieved_products = set()

    for doc in retrieved_docs:

        product = doc.metadata.get(
            "product_name"
        )

        if product:
            retrieved_products.add(
                str(product)
                .strip()
                .lower()
            )

    checked = 0
    grounded = 0

    for row in rows:

        product = row[
            "Product Name"
        ].strip().lower()

        if not product:
            continue

        checked += 1

        if product in retrieved_products:
            grounded += 1

    rate = (
        grounded / checked
        if checked
        else 0.0
    )

    return {
        "checked": checked,
        "grounded": grounded,
        "rate": rate,
    }


# ============================================================
# Check 8: Unsupported Claims
#
# Conservative check:
# generated Value must occur in the evidence for
# the same product.
# ============================================================

def check_unsupported_claims(
    rows,
    retrieved_docs,
):

    evidence_by_product = {}

    for doc in retrieved_docs:

        metadata = doc.metadata

        product = str(
            metadata.get(
                "product_name",
                "",
            )
        ).strip().lower()

        snippet = str(
            metadata.get(
                "source_snippet",
                "",
            )
        ).strip()

        if not product or not snippet:
            continue

        evidence_by_product.setdefault(
            product,
            "",
        )

        evidence_by_product[
            product
        ] += " " + snippet

    checked = 0
    unsupported = 0

    unsupported_examples = []

    for row in rows:

        product = row[
            "Product Name"
        ].strip().lower()

        value = row[
            "Value"
        ].strip()

        if not product or not value:
            continue

        checked += 1

        evidence = evidence_by_product.get(
            product,
            "",
        ).lower()

        # Remove markdown links if any.
        clean_value = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            value,
        )

        if (
            clean_value.lower()
            not in evidence
        ):

            unsupported += 1

            unsupported_examples.append(
                {
                    "product":
                        row["Product Name"],
                    "feature":
                        row["Feature"],
                    "value":
                        row["Value"],
                }
            )

    unsupported_rate = (
        unsupported / checked
        if checked
        else 0.0
    )

    return {
        "checked": checked,
        "unsupported": unsupported,
        "rate": unsupported_rate,
        "examples":
            unsupported_examples[:5],
    }


# ============================================================
# Evaluate One Question
# ============================================================

def evaluate_question(test_case):

    question = test_case["question"]

    print("\n" + "=" * 80)
    print(
        f"{test_case['id']}: {question}"
    )

    try:

        # ----------------------------------------------------
        # 1. Create query plan
        # ----------------------------------------------------

        plan = plan_query(
            question
        )

        # ----------------------------------------------------
        # 2. Retrieve
        #
        # YOUR retrieve() REQUIRES plan
        # ----------------------------------------------------

        retrieved_docs = retrieve(
            question,
            plan,
            k=10,
            vector_k=10,
            keyword_k=10,
        )

        # ----------------------------------------------------
        # 3. Generate answer
        #
        # ask() internally creates its own plan/retrieval.
        # ----------------------------------------------------

        result = ask(
            question
        )

        answer = result[
            "answer"
        ]

        # ----------------------------------------------------
        # 4. Parse generated table
        # ----------------------------------------------------

        rows = parse_table(
            answer
        )

        # ----------------------------------------------------
        # 5. Automated checks
        # ----------------------------------------------------

        retrieval_hit = check_retrieval_hit(
            test_case,
            retrieved_docs,
        )

        format_ok = check_output_format(
            answer
        )

        citations_ok = check_citations_present(
            rows
        )

        urls_ok = check_urls_present(
            rows
        )

        relevance_values_ok = (
            check_relevance_values(
                rows
            )
        )

        citation_grounding = (
            check_citation_grounding(
                rows,
                retrieved_docs,
            )
        )

        product_grounding = (
            check_product_grounding(
                rows,
                retrieved_docs,
            )
        )

        unsupported = (
            check_unsupported_claims(
                rows,
                retrieved_docs,
            )
        )

        # ----------------------------------------------------
        # 6. Print result
        # ----------------------------------------------------

        print(
            "Query Plan:",
            plan.model_dump(),
        )

        print(
            "Retrieved documents:",
            len(retrieved_docs),
        )

        print(
            "Retrieval Hit:",
            retrieval_hit,
        )

        print(
            "Format OK:",
            format_ok,
        )

        print(
            "Citations Present:",
            citations_ok,
        )

        print(
            "URLs Present:",
            urls_ok,
        )

        print(
            "Citation Grounding:",
            f"{citation_grounding['rate'] * 100:.2f}%",
        )

        print(
            "Product Grounding:",
            f"{product_grounding['rate'] * 100:.2f}%",
        )

        print(
            "Unsupported Claim Rate:",
            f"{unsupported['rate'] * 100:.2f}%",
        )

        return {
            "id":
                test_case["id"],

            "question":
                question,

            "query_plan":
                plan.model_dump(),

            "retrieved_documents":
                len(retrieved_docs),

            "retrieval_hit":
                retrieval_hit,

            "format_ok":
                format_ok,

            "citations_present":
                citations_ok,

            "urls_present":
                urls_ok,

            "relevance_values_valid":
                relevance_values_ok,

            "citation_grounding_rate":
                citation_grounding[
                    "rate"
                ],

            "product_grounding_rate":
                product_grounding[
                    "rate"
                ],

            "unsupported_claim_rate":
                unsupported[
                    "rate"
                ],

            "unsupported_claims":
                unsupported[
                    "unsupported"
                ],

            "rag_relevance":
                result.get(
                    "relevance"
                ),

            "rag_relevance_reason":
                result.get(
                    "relevance_reason"
                ),

            "answer":
                answer,

            "unsupported_examples":
                unsupported[
                    "examples"
                ],
        }

    except Exception as exc:

        print(
            "ERROR:",
            type(exc).__name__,
            str(exc),
        )

        return {
            "id":
                test_case["id"],

            "question":
                question,

            "error":
                f"{type(exc).__name__}: {exc}",

            "retrieval_hit":
                False,

            "format_ok":
                False,

            "citations_present":
                False,

            "urls_present":
                False,

            "relevance_values_valid":
                False,

            "citation_grounding_rate":
                0.0,

            "product_grounding_rate":
                0.0,

            "unsupported_claim_rate":
                1.0,

            "unsupported_claims":
                0,

        }


# ============================================================
# Calculate Overall Metrics
# ============================================================

def calculate_metrics(results):

    total = len(results)

    if total == 0:
        return {}

    def bool_rate(key):

        return (
            sum(
                bool(
                    result.get(
                        key,
                        False,
                    )
                )
                for result in results
            )
            / total
            * 100
        )

    def avg_rate(key):

        return (
            sum(
                result.get(
                    key,
                    0.0,
                )
                for result in results
            )
            / total
            * 100
        )

    return {

        "test_questions":
            total,

        "retrieval_hit_rate":
            round(
                bool_rate(
                    "retrieval_hit"
                ),
                2,
            ),

        "format_compliance":
            round(
                bool_rate(
                    "format_ok"
                ),
                2,
            ),

        "citation_presence_rate":
            round(
                bool_rate(
                    "citations_present"
                ),
                2,
            ),

        "product_url_rate":
            round(
                bool_rate(
                    "urls_present"
                ),
                2,
            ),

        "relevance_value_rate":
            round(
                bool_rate(
                    "relevance_values_valid"
                ),
                2,
            ),

        "citation_grounding_rate":
            round(
                avg_rate(
                    "citation_grounding_rate"
                ),
                2,
            ),

        "product_grounding_rate":
            round(
                avg_rate(
                    "product_grounding_rate"
                ),
                2,
            ),

        "unsupported_claim_rate":
            round(
                avg_rate(
                    "unsupported_claim_rate"
                ),
                2,
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    results = []

    for test_case in TEST_CASES:

        result = evaluate_question(
            test_case
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    print("\n")
    print("=" * 80)
    print("RAG EVALUATION SUMMARY")
    print("=" * 80)

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    output = {
        "metrics":
            metrics,
        "results":
            results,
    }

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nDetailed results saved to "
        "evaluation_results.json"
    )


if __name__ == "__main__":
    main()