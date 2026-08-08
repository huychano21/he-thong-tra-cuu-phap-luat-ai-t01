#!/usr/bin/env python3
"""Hybrid retrieval over a reproducible subset of Vietnamese legal documents."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
MODEL_MAX_SEQ_LENGTH = 8192
DOCUMENT_BATCH_SIZE = 4
BENCHMARK_WARMUP_PASSES = 1
BENCHMARK_MEASURED_PASSES = 5
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_corpus(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    required = {"id", "title", "text", "source"}
    if not rows or any(required - row.keys() for row in rows):
        raise ValueError("Corpus rỗng hoặc thiếu id/title/text/source")
    return rows


def load_queries(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    required = {"qid", "question", "relevant_ids"}
    if not rows or any(required - row.keys() for row in rows):
        raise ValueError("Queries rỗng hoặc thiếu qid/question/relevant_ids")
    return rows


def rewrite_query(history: str, question: str) -> str:
    original = question.strip()

    if not history.strip():
        return original

    try:
        base_url = os.environ.get(
            "LLM_BASE_URL",
            "https://openrouter.ai/api/v1",
        ).rstrip("/")
        api_key = os.environ["LLM_API_KEY"]
        model = os.environ["LLM_MODEL"]

        system_prompt = """
Bạn viết lại câu hỏi hội thoại để phục vụ truy xuất tài liệu.

Yêu cầu:

- Dùng lịch sử hội thoại để làm rõ các tham chiếu mơ hồ.
- Chỉ sử dụng thông tin có trong lịch sử và câu hỏi.
- Không tự thêm tên người, tổ chức, thời gian hoặc điều kiện mới.
- Nếu câu hỏi đã tự đủ nghĩa, giữ nguyên câu hỏi.
- Chỉ trả về câu hỏi đã viết lại, không giải thích.
""".strip()

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Lịch sử hội thoại:\n{history.strip()}\n\n"
                            f"Câu hỏi mới:\n{original}"
                        ),
                    },
                ],
            },
            timeout=30,
        )
        response.raise_for_status()

        rewritten = response.json()["choices"][0]["message"]["content"].strip()
        return rewritten or original
    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        AttributeError,
    ):
        return original


def ranked_ids(scores: np.ndarray) -> list[int]:
    return np.argsort(-scores, kind="stable").tolist()


def rrf(
    rankings: list[list[int]], weights: list[float], k: int = 60
) -> tuple[list[int], np.ndarray]:
    if (
        len(rankings) != len(weights)
        or k < 1
        or any(weight < 0 for weight in weights)
        or not any(weights)
    ):
        raise ValueError("rankings/weights không hợp lệ")
    size = max(max(order) for order in rankings) + 1
    scores = np.zeros(size, dtype=float)
    for order, weight in zip(rankings, weights, strict=True):
        for rank, doc_index in enumerate(order, start=1):
            scores[doc_index] += weight / (k + rank)
    return ranked_ids(scores), scores


class HybridRetriever:
    def __init__(self, corpus_path: Path, model_name: str = MODEL_NAME):
        self.rows = load_corpus(corpus_path)
        self.texts = [f"{row['title']}. {row['text']}" for row in self.rows]
        self.bm25 = BM25Okapi([tokenize(text) for text in self.texts])
        self.model = SentenceTransformer(
            model_name, local_files_only=os.getenv("HF_HUB_OFFLINE") == "1"
        )
        self.model.max_seq_length = MODEL_MAX_SEQ_LENGTH
        tokenized = self.model.tokenizer(
            self.texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
        self.corpus_token_lengths = list(map(int, tokenized["length"]))
        encode_documents = getattr(self.model, "encode_document", self.model.encode)
        self.embeddings = encode_documents(
            self.texts,
            batch_size=DOCUMENT_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> dict[str, object]:
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")

        started = time.perf_counter()
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        bm25_order = ranked_ids(bm25_scores)
        bm25_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        encode_query = getattr(self.model, "encode_query", self.model.encode)
        query_embedding = encode_query(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        dense_scores = np.asarray(self.embeddings @ query_embedding)
        dense_order = ranked_ids(dense_scores)
        dense_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        hybrid_order, hybrid_scores = rrf(
            [bm25_order, dense_order], [bm25_weight, dense_weight], rrf_k
        )
        fusion_ms = (time.perf_counter() - started) * 1000

        def pack(order: list[int], scores: np.ndarray) -> list[dict[str, object]]:
            return [
                {
                    "rank": rank,
                    "id": self.rows[index]["id"],
                    "title": self.rows[index]["title"],
                    "score": round(float(scores[index]), 6),
                    "source": self.rows[index]["source"],
                }
                for rank, index in enumerate(order[:top_k], start=1)
            ]

        return {
            "bm25": pack(bm25_order, bm25_scores),
            "dense": pack(dense_order, dense_scores),
            "hybrid": pack(hybrid_order, hybrid_scores),
            "latency_ms": {
                "bm25": round(bm25_ms, 2),
                "dense": round(dense_ms, 2),
                "fusion": round(fusion_ms, 2),
            },
        }


def evaluate_ranking(
    retrieved: list[str], relevant: set[str], cutoffs: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    first_rank = next(
        (rank for rank, doc_id in enumerate(retrieved, start=1) if doc_id in relevant),
        None,
    )
    metrics = {
        f"Recall@{k}": len(relevant.intersection(retrieved[:k])) / len(relevant)
        for k in cutoffs
    }
    metrics["MRR@10"] = 1 / first_rank if first_rank and first_rank <= 10 else 0.0
    dcg = sum(
        1 / np.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:10], start=1)
        if doc_id in relevant
    )
    ideal = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, min(len(relevant), 10) + 1)
    )
    metrics["nDCG@10"] = dcg / ideal
    return metrics


def benchmark(
    retriever: HybridRetriever,
    queries_path: Path,
    output_dir: Path,
    warmup_passes: int = BENCHMARK_WARMUP_PASSES,
    measured_passes: int = BENCHMARK_MEASURED_PASSES,
    **search_options: object,
) -> None:
    if warmup_passes < 0 or measured_passes < 1:
        raise ValueError("warmup_passes/measured_passes không hợp lệ")

    queries = load_queries(queries_path)
    modes = ("bm25", "dense", "hybrid")
    metric_names = ("Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")
    totals = {mode: {name: [] for name in metric_names} for mode in modes}
    latency = {mode: [] for mode in modes}
    per_query: list[dict[str, object]] = []
    wins = 0
    search_options["top_k"] = 10

    for _ in range(warmup_passes):
        for query_row in queries:
            retriever.search(str(query_row["question"]), **search_options)

    for measured_pass in range(measured_passes):
        for query_row in queries:
            result = retriever.search(str(query_row["question"]), **search_options)
            latency["bm25"].append(result["latency_ms"]["bm25"])
            latency["dense"].append(result["latency_ms"]["dense"])
            latency["hybrid"].append(
                result["latency_ms"]["bm25"]
                + result["latency_ms"]["dense"]
                + result["latency_ms"]["fusion"]
            )

            if measured_pass > 0:
                continue

            relevant = set(map(str, query_row["relevant_ids"]))
            row: dict[str, object] = {
                "qid": query_row["qid"],
                "question": query_row["question"],
            }
            ranks: dict[str, int] = {}
            for mode in modes:
                ids = [str(item["id"]) for item in result[mode]]
                metrics = evaluate_ranking(ids, relevant)
                for name, value in metrics.items():
                    totals[mode][name].append(value)
                    row[f"{mode}_{name}"] = round(value, 6)
                ranks[mode] = next(
                    (
                        rank
                        for rank, doc_id in enumerate(ids, start=1)
                        if doc_id in relevant
                    ),
                    99,
                )
            row["dense_first_relevant_rank"] = ranks["dense"]
            row["hybrid_first_relevant_rank"] = ranks["hybrid"]
            wins += ranks["hybrid"] < ranks["dense"]
            per_query.append(row)

    memory_gb = None
    if {"SC_PAGE_SIZE", "SC_PHYS_PAGES"}.issubset(os.sysconf_names):
        memory_gb = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3,
            2,
        )

    report = {
        "documents": len(retriever.rows),
        "queries": len(queries),
        "embedding_model": {
            "name": MODEL_NAME,
            "dimension": int(retriever.embeddings.shape[1]),
            "max_seq_length": retriever.model.max_seq_length,
            "corpus_token_length": {
                "mean": round(float(np.mean(retriever.corpus_token_lengths)), 2),
                "p95": round(
                    float(np.percentile(retriever.corpus_token_lengths, 95)), 2
                ),
                "max": max(retriever.corpus_token_lengths),
                "over_limit": sum(
                    length > retriever.model.max_seq_length
                    for length in retriever.corpus_token_lengths
                ),
                "over_limit_rate": round(
                    float(
                        np.mean(
                            np.asarray(retriever.corpus_token_lengths)
                            > retriever.model.max_seq_length
                        )
                    ),
                    4,
                ),
            },
        },
        "benchmark_protocol": {
            "warmup_passes": warmup_passes,
            "measured_passes": measured_passes,
            "queries_per_pass": len(queries),
            "latency_samples_per_retriever": len(queries) * measured_passes,
            "metrics_evaluated_passes": 1,
        },
        "environment": {
            "operating_system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "memory_gb": memory_gb,
            "python_version": platform.python_version(),
            "pytorch_version": str(torch.__version__),
            "compute_device": str(retriever.model.device),
        },
        "metrics": {
            mode: {
                name: round(float(np.mean(values)), 4)
                for name, values in totals[mode].items()
            }
            for mode in modes
        },
        "latency_ms": {
            mode: {
                "p50": round(float(np.percentile(values, 50)), 2),
                "p95": round(float(np.percentile(values, 95)), 2),
            }
            for mode, values in latency.items()
        },
        "hybrid_better_than_dense_queries": wins,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def chat(retriever: HybridRetriever, **search_options: object) -> None:
    """Retrieval-only chatbot: show evidence instead of inventing legal advice."""
    history = ""
    print("Gõ /reset để xóa ngữ cảnh, /quit để thoát.")
    while True:
        try:
            question = input("Bạn: ").strip()
        except EOFError:
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/reset":
            history = ""
            print("Đã xóa ngữ cảnh.")
            continue

        rewritten = rewrite_query(history, question)
        result = retriever.search(rewritten, **search_options)
        print(json.dumps(
            {"original": question, "rewritten": rewritten},
            ensure_ascii=False,
        ))
        print("Trợ lý: Tôi tìm thấy các tài liệu gần nhất:")
        for item in result["hybrid"]:
            print(f"- {item['id']}: {item['title']} | {item['source']}")
        print("Đây là kết quả truy xuất, không phải kết luận pháp lý.")
        history = rewritten


def self_check() -> None:
    assert tokenize("Đăng ký doanh nghiệp!") == ["đăng", "ký", "doanh", "nghiệp"]
    with patch.dict(os.environ, {}, clear=True):
        rewritten = rewrite_query(
            "thủ tục đăng ký doanh nghiệp qua mạng", "Thủ tục này thực hiện ở đâu?"
        )
    assert rewritten == "Thủ tục này thực hiện ở đâu?"
    order, _ = rrf([[0, 1], [1, 0]], [2.0, 1.0], k=60)
    assert order[0] == 0
    metrics = evaluate_ranking(["b", "a"], {"a"})
    assert metrics["Recall@1"] == 0 and metrics["MRR@10"] == 0.5
    print("self-check: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--history", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    retriever = HybridRetriever(Path(__file__).with_name("corpus.jsonl"))
    search_options = {
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "bm25_weight": args.bm25_weight,
        "dense_weight": args.dense_weight,
    }
    if args.chat:
        chat(retriever, **search_options)
        return
    if args.benchmark:
        benchmark(
            retriever,
            Path(__file__).with_name("queries.jsonl"),
            Path(__file__).with_name("evaluation"),
            **search_options,
        )
        return
    query = rewrite_query(args.history, args.query or "đăng ký doanh nghiệp qua mạng")
    print(json.dumps({"original": args.query, "rewritten": query}, ensure_ascii=False))
    print(json.dumps(retriever.search(query, **search_options), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
