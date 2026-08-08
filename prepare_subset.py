#!/usr/bin/env python3
"""Build a deterministic retrieval subset from YuITC test.parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

DATASET_ID = "YuITC/Vietnamese-Legal-Documents"
REVISION = "a5195e351b08074529795283c77b4c9787c727f9"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
DOWNLOAD_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/test.parquet"
)


def unique_contexts(row: object) -> list[tuple[str, str]]:
    return list(dict.fromkeys(
        (str(int(cid)), str(text))
        for cid, text in zip(row.cid, row.context_list, strict=True)
    ))


def title(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:100] or "Văn bản pháp luật"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(source: Path, root: Path) -> None:
    frame = pd.read_parquet(source)
    unique_count = frame.cid.map(lambda values: len(set(map(int, values))))
    selected = pd.concat([
        frame[unique_count == 1].head(20),
        frame[unique_count > 1].head(10),
    ]).sort_index()

    corpus: dict[str, str] = {}
    for row in selected.itertuples():
        corpus.update(unique_contexts(row))
    for row in frame.itertuples():
        corpus.update(unique_contexts(row))
        if len(corpus) >= 120:
            break
    corpus = dict(list(corpus.items())[:120])

    corpus_rows = [
        {
            "id": cid,
            "title": f"CID {cid}: {title(text)}",
            "text": text,
            "source": DATASET_URL,
        }
        for cid, text in corpus.items()
    ]
    query_rows = [
        {
            "qid": str(int(row.qid)),
            "question": str(row.question),
            "relevant_ids": [cid for cid, _ in unique_contexts(row)],
        }
        for row in selected.itertuples()
    ]
    if any(cid not in corpus for row in query_rows for cid in row["relevant_ids"]):
        raise ValueError("Subset corpus thiếu tài liệu ground truth")

    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(queries_path, query_rows)
    manifest = {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "license": "MIT",
        "split": "test",
        "selection": "first 20 single-relevant + first 10 multi-relevant; 120 documents",
        "query_count": len(query_rows),
        "document_count": len(corpus_rows),
        "query_ids": [row["qid"] for row in query_rows],
        "corpus_sha256": sha256(corpus_path),
        "queries_sha256": sha256(queries_path),
    }
    (root / "subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(corpus_rows)} documents and {len(query_rows)} queries")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    source = args.source or root / "data" / "test.parquet"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DOWNLOAD_URL, source)
    build(source, root)


if __name__ == "__main__":
    main()
