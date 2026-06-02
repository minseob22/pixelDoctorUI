# test_rag.py

from rag.ingest import load_pdf

from rag.chunker import chunck_documents

from rag.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store
)

from rag.retriever import (
    retrieve_relevant_chunks
)


# =========================================
# 최초 1회만 True
# =========================================

BUILD_VECTOR_DB = True


# =========================================
# Vector DB 생성 및 저장
# =========================================

if BUILD_VECTOR_DB:

    print("\n[INFO] Building vector DB...\n")

    pdf_path = (
        "papers/raw_pdfs/0068KJR_kjr-26-900.pdf"
    )

    documents = load_pdf(pdf_path)

    print(
        f"[INFO] Loaded {len(documents)} pages"
    )

    chunks = chunck_documents(documents)

    print(
        f"[INFO] Generated {len(chunks)} chunks"
    )

    vector_store = build_vector_store(
        chunks
    )

    save_vector_store(vector_store)

    print(
        "\n[INFO] Vector DB build complete.\n"
    )


# =========================================
# 저장된 Vector DB 로드
# =========================================

print("\n[INFO] Loading vector DB...\n")

vector_store = load_vector_store()

print("\n[INFO] Vector DB loaded.\n")


# =========================================
# Query
# =========================================

query = """
radiomics tumor heterogeneity
texture complexity
surface irregularity
"""


# =========================================
# Retrieval
# =========================================

results = retrieve_relevant_chunks(
    vector_store,
    query,
    k=3
)


# =========================================
# 출력
# =========================================

for i, doc in enumerate(results):

    print(
        f"\n===== RESULT {i+1} =====\n"
    )

    print(doc.page_content)