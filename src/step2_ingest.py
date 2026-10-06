import os
import shutil
import pandas as pd
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


load_dotenv()


# ============================================================
# Configuration
# ============================================================

PHILIPS_FILE = "./data/philips/philips_products_flat.csv"
ORAL_B_FILE = "./data/oralb/oralb_products_flat.csv"

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "toothbrush_products"


# ============================================================
# CSV columns
# ============================================================

REQUIRED_COLUMNS = [
    "brand",
    "product_name",
    "price_usd",
    "product_category",
    "customer_rating",
    "customer_rating_count",
    "app_connectivity",
    "plaque_claims",
    "gum_claims",
    "brush_modes_count",
    "short_description",
    "long_description",
    "product_url",
]


# ============================================================
# Attributes that will become individual RAG documents
# ============================================================

ATTRIBUTE_COLUMNS = [
    "price_usd",
    "product_category",
    "customer_rating",
    "customer_rating_count",
    "app_connectivity",
    "plaque_claims",
    "gum_claims",
    "brush_modes_count",
    "short_description",
    "long_description",
]


# ============================================================
# Recursive splitter for long descriptions
# ============================================================

long_description_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        ""
    ],
    length_function=len,
)


# ============================================================
# Helpers
# ============================================================

def clean_value(value):
    """
    Convert NaN/empty values to None.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def make_source_id(product_name):
    """
    Convert product name into a clean source ID.

    Example:
        Philips Sonicare 4100
        ->
        philips_sonicare_4100
    """

    source_id = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        product_name.strip()
    )

    return source_id.strip("_").lower()


def load_csv(file_path):
    """
    Load CSV and normalize column names.
    """

    df = pd.read_csv(file_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def validate_columns(df, file_path):
    """
    Make sure the CSV contains all required columns.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\n{file_path} is missing columns:\n"
            f"{missing_columns}"
        )


# ============================================================
# Create Documents
# ============================================================

def create_documents_from_row(row):

    documents = []

    brand = clean_value(row["brand"])
    product_name = clean_value(row["product_name"])
    product_url = clean_value(row["product_url"])

    # Product-level information
    price_usd = clean_value(row["price_usd"])
    customer_rating = clean_value(row["customer_rating"])
    customer_rating_count = clean_value(
        row["customer_rating_count"]
    )

    # Skip rows without product name
    if not product_name:
        return documents

    # Product-level source ID
    source_id = make_source_id(product_name)

    # --------------------------------------------------------
    # Create one document per attribute
    # --------------------------------------------------------

    for attribute in ATTRIBUTE_COLUMNS:

        value = clean_value(row[attribute])

        # Don't create documents for missing fields
        if value is None:
            continue

        # ====================================================
        # LONG DESCRIPTION
        # ====================================================

        if attribute == "long_description":

            chunks = long_description_splitter.split_text(
                value
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                document_id = (
                    f"{source_id}_"
                    f"{attribute}_"
                    f"{chunk_number}"
                )

                content = f"""
Brand: {brand}

Product: {product_name}

Attribute: Long Description

Value:
{chunk}
""".strip()

                document = Document(
                    page_content=content,

                    metadata={
                        # IDs
                        "source_id": source_id,
                        "document_id": document_id,

                        # Product information
                        "brand": brand,
                        "product_name": product_name,

                        # Product-level metadata
                        "price_usd": price_usd,
                        "customer_rating": customer_rating,
                        "customer_rating_count":
                            customer_rating_count,

                        # Current attribute
                        "attribute": attribute,

                        # Chunk information
                        "chunk_id": chunk_number,

                        # Only the actual chunk is used
                        # as source evidence
                        "source_snippet": chunk,

                        # Source information
                        "source_url": product_url,
                        "source_type":
                            "official_us_product_page",
                    }
                )

                documents.append(document)

            continue

        # ====================================================
        # NORMAL ATTRIBUTES
        # ====================================================

        document_id = (
            f"{source_id}_{attribute}"
        )

        content = f"""
Brand: {brand}

Product: {product_name}

Attribute: {attribute}

Value:
{value}
""".strip()

        document = Document(
            page_content=content,

            metadata={
                # IDs
                "source_id": source_id,
                "document_id": document_id,

                # Product information
                "brand": brand,
                "product_name": product_name,

                # Product-level metadata
                "price_usd": price_usd,
                "customer_rating": customer_rating,
                "customer_rating_count":
                    customer_rating_count,

                # Current attribute
                "attribute": attribute,

                # Source information
                "source_url": product_url,

                # For short attributes, the entire
                # value is the source snippet
                "source_snippet": value,

                "source_type":
                    "official_us_product_page",
            }
        )

        documents.append(document)

    return documents


# ============================================================
# Main ingestion process
# ============================================================

def main():

    print("\nLoading CSV files...")

    # --------------------------------------------------------
    # Load Philips
    # --------------------------------------------------------

    philips_df = load_csv(
        PHILIPS_FILE
    )

    validate_columns(
        philips_df,
        PHILIPS_FILE
    )

    # --------------------------------------------------------
    # Load Oral-B
    # --------------------------------------------------------

    oralb_df = load_csv(
        ORAL_B_FILE
    )

    validate_columns(
        oralb_df,
        ORAL_B_FILE
    )

    print(
        f"Philips products: {len(philips_df)}"
    )

    print(
        f"Oral-B products: {len(oralb_df)}"
    )

    # --------------------------------------------------------
    # Create Documents
    # --------------------------------------------------------

    documents = []

    # Philips
    for _, row in philips_df.iterrows():

        documents.extend(
            create_documents_from_row(row)
        )

    # Oral-B
    for _, row in oralb_df.iterrows():

        documents.extend(
            create_documents_from_row(row)
        )

    print(
        f"Total RAG documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # Show example
    # --------------------------------------------------------

    if documents:

        print("\nExample document:")
        print("--------------------------------")

        print(
            documents[0].page_content
        )

        print("\nMetadata:")

        print(
            documents[0].metadata
        )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Remove previous Chroma database
    # --------------------------------------------------------

    if os.path.exists(CHROMA_DIR):

        print(
            "\nRemoving existing Chroma database..."
        )

        shutil.rmtree(CHROMA_DIR)

    # --------------------------------------------------------
    # Create Chroma
    # --------------------------------------------------------

    print(
        "\nCreating Chroma vector database..."
    )

    Chroma.from_documents(

        documents=documents,

        embedding=embeddings,

        collection_name=COLLECTION_NAME,

        persist_directory=CHROMA_DIR,
    )

    print(
        "\nChroma database created successfully."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()