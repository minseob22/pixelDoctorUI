from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chat_graph.retriever import evaluate_retrieval_quality, retrieve_documents


def main() -> None:
    query = "TB diagnosis on chest X-ray"
    docs = retrieve_documents(query, top_k=3)
    quality, reason = evaluate_retrieval_quality(docs)
    retriever_types = sorted(
        {doc.get("retriever", "unknown") for doc in docs if isinstance(doc, dict)}
    )

    print(f"[warmup_faiss] query={query}")
    print(f"[warmup_faiss] retrieved_docs_count={len(docs)}")
    print(f"[warmup_faiss] retriever_types={retriever_types}")
    print(f"[warmup_faiss] retrieval_quality={quality}")
    print(f"[warmup_faiss] reason={reason}")
    if docs:
        print(f"[warmup_faiss] first_source={docs[0].get('source')}")


if __name__ == "__main__":
    main()
