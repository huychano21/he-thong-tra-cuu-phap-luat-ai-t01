#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hệ thống AI Tra cứu Văn bản Pháp luật và Quy chế Đào tạo Thông minh
Ứng dụng Thuật toán Tìm kiếm Lai (Hybrid Search: BM25, Dense Retrieval và RRF)
kết hợp Mô hình Ngôn ngữ Lớn cục bộ (Local LLM Ollama Qwen 2.5 7B).

Phiên bản nâng cấp Vòng 5:
1. Nâng cấp toàn diện Mô-đun Chuẩn hóa Câu hỏi Pháp lý (`rewrite_query_with_trace`):
   - Hoạt động ngay từ lượt tra cứu đầu tiên (kể cả khi chưa có lịch sử hội thoại):
     Chuyển đổi từ viết tắt, khẩu ngữ đời thường (VD: "nd 168", "uống rượu bia", "vượt đèn đỏ",
     "bằng lái", "vphc", "luật anm") sang thuật ngữ pháp điển chính quy.
   - Tự động cập nhật theo các văn bản luật mới nạp vào kho (`build_corpus_catalog_hint`):
     Khi người dùng nạp văn bản mới (như Nghị định 168/2024/NĐ-CP, Luật 116/2025/QH15...), hệ thống
     tự động trích xuất số hiệu, tên văn bản và danh mục điều khoản để ánh xạ câu hỏi chuẩn hóa
     và tự động sinh các câu hỏi gợi ý sát với văn bản mới.
2. Giữ tương thích ngược 100% với `python hybrid_search.py --self-check` và `--benchmark`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import requests
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from pyvi import ViTokenizer  # type: ignore

    HAS_PYVI = True
except ImportError:
    ViTokenizer = None
    HAS_PYVI = False

try:
    import torch_directml  # type: ignore

    HAS_DIRECTML = True
except ImportError:
    torch_directml = None
    HAS_DIRECTML = False

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
MODEL_MAX_SEQ_LENGTH = 8192
BENCHMARK_WARMUP_PASSES = 1
BENCHMARK_MEASURED_PASSES = 5

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_API_KEY = "ollama"
DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_LLM_TIMEOUT = 12

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

FALLBACK_COMPOUND_TERMS = (
    "học viện an ninh nhân dân",
    "an ninh nhân dân",
    "công an nhân dân",
    "bộ công an",
    "vi phạm hành chính",
    "xử phạt vi phạm hành chính",
    "thời hiệu xử phạt",
    "đăng ký doanh nghiệp",
    "chế độ tiền lương",
    "phụ cấp thâm niên",
    "quy chế đào tạo",
    "văn bản quy phạm pháp luật",
    "nhận hối lộ",
    "ngân hàng chính sách xã hội",
    "an ninh mạng",
    "giao thông đường bộ",
    "giấy phép lái xe",
    "nồng độ cồn",
)

# Từ điển chuẩn hóa viết tắt & khẩu ngữ đời thường sang thuật ngữ pháp điển Việt Nam
LEGAL_ABBREVIATION_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvphc\b", re.IGNORECASE), "vi phạm hành chính"),
    (re.compile(r"\bgplx\b", re.IGNORECASE), "giấy phép lái xe"),
    (re.compile(r"\bcand\b", re.IGNORECASE), "Công an nhân dân"),
    (re.compile(r"\bannd\b", re.IGNORECASE), "An ninh nhân dân"),
    (re.compile(r"\bt01\b", re.IGNORECASE), "Học viện An ninh nhân dân"),
    (re.compile(r"\blhs\b|\bblhs\b", re.IGNORECASE), "Bộ luật Hình sự"),
    (re.compile(r"\banm\b", re.IGNORECASE), "an ninh mạng"),
    (re.compile(r"\bgtđb\b|\bgtdb\b", re.IGNORECASE), "giao thông đường bộ"),
    (re.compile(r"\bnsnn\b", re.IGNORECASE), "ngân sách nhà nước"),
    (re.compile(r"\bbhxh\b", re.IGNORECASE), "bảo hiểm xã hội"),
    (re.compile(r"\bbhyt\b", re.IGNORECASE), "bảo hiểm y tế"),
    (re.compile(r"\bvksnd\b", re.IGNORECASE), "Viện kiểm sát nhân dân"),
    (re.compile(r"\btand\b", re.IGNORECASE), "Tòa án nhân dân"),
]

# Từ điển mở rộng cụm từ đời thường sang thuật ngữ trong điều khoản luật (giữ cả ý gốc lẫn thuật ngữ pháp lý)
COLLOQUIAL_LEGAL_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(?:uống\s+rượu\s+bia|uống\s+bia\s+rượu|uống\s+rượu|uống\s+bia|say\s+rượu|có\s+cồn)\b", re.IGNORECASE),
        "trong máu hoặc hơi thở có nồng độ cồn",
    ),
    (
        re.compile(r"\b(?:bằng\s+lái\s+xe|bằng\s+lái)\b", re.IGNORECASE),
        "giấy phép lái xe",
    ),
    (
        re.compile(r"\b(?:vượt\s+đèn\s+đỏ|vượt\s+đèn\s+vàng)\b", re.IGNORECASE),
        "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    ),
    (
        re.compile(r"\b(?:bấm\s+điện\s+thoại|nghe\s+điện\s+thoại|xài\s+điện\s+thoại)\b", re.IGNORECASE),
        "sử dụng điện thoại",
    ),
    (
        re.compile(r"\b(?:che\s+ô|cầm\s+ô|bung\s+dù)\b", re.IGNORECASE),
        "sử dụng ô (dù)",
    ),
    (
        re.compile(r"\b(?:đi\s+hàng\s+ngang|chạy\s+hàng\s+ngang)\b", re.IGNORECASE),
        "đi dàn hàng ngang",
    ),
    (
        re.compile(r"\b(?:lập\s+công\s+ty|mở\s+công\s+ty)\b", re.IGNORECASE),
        "đăng ký doanh nghiệp",
    ),
]


def get_app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _probe_nvidia_gpu_via_smi() -> str | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def detect_hardware_accelerator() -> dict[str, Any]:
    forced_device = os.environ.get("FORCE_COMPUTE_DEVICE", "").strip().lower()
    cpu_threads = max(2, (os.cpu_count() or 4))
    torch_ver = str(getattr(torch, "__version__", ""))

    try:
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(min(12, cpu_threads))
    except Exception:
        pass

    if forced_device == "cpu":
        return {
            "device": "cpu",
            "backend": "CPU Multi-Threading",
            "device_name": f"CPU ({cpu_threads} luồng xử lý)",
            "batch_size": 8,
            "use_fp16": False,
            "gpu_available": False,
            "pytorch_version": torch_ver,
            "cuda_mismatch_warning": None,
        }

    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
            return {
                "device": "cuda",
                "backend": "NVIDIA CUDA GPU",
                "device_name": f"{dev_name} ({vram_gb} GB VRAM)",
                "batch_size": 32,
                "use_fp16": True,
                "gpu_available": True,
                "pytorch_version": torch_ver,
                "cuda_mismatch_warning": None,
            }
    except Exception:
        pass

    physical_nvidia = _probe_nvidia_gpu_via_smi()
    cuda_warning = None
    if physical_nvidia:
        cuda_warning = (
            f"Phát hiện card rời '{physical_nvidia}' trên máy nhưng PyTorch trong .venv đang là bản CPU ({torch_ver})! "
            f"Hãy chạy: python hybrid_search.py --fix-gpu"
        )

    if HAS_DIRECTML and torch_directml is not None:
        try:
            dml_device = torch_directml.device()
            dml_name = torch_directml.device_name(0)
            return {
                "device": str(dml_device),
                "backend": "DirectML (GPU tích hợp Intel Iris / AMD)",
                "device_name": dml_name,
                "batch_size": 16,
                "use_fp16": False,
                "gpu_available": True,
                "pytorch_version": torch_ver,
                "cuda_mismatch_warning": cuda_warning,
            }
        except Exception:
            pass

    try:
        if (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return {
                "device": "mps",
                "backend": "Apple Metal GPU (MPS)",
                "device_name": "Apple Silicon GPU",
                "batch_size": 16,
                "use_fp16": False,
                "gpu_available": True,
                "pytorch_version": torch_ver,
                "cuda_mismatch_warning": None,
            }
    except Exception:
        pass

    return {
        "device": "cpu",
        "backend": "CPU Đa luồng" + (" (Chưa bật CUDA)" if physical_nvidia else ""),
        "device_name": (
            f"Có {physical_nvidia} nhưng đang chạy CPU ({cpu_threads} luồng)"
            if physical_nvidia
            else f"{platform.processor() or 'CPU tiêu chuẩn'} ({cpu_threads} luồng)"
        ),
        "batch_size": 8,
        "use_fp16": False,
        "gpu_available": False,
        "pytorch_version": torch_ver,
        "cuda_mismatch_warning": cuda_warning,
    }


def fix_pytorch_cuda_installation() -> None:
    print("[*] Đang kiểm tra card đồ họa NVIDIA trên máy...")
    gpu_info = _probe_nvidia_gpu_via_smi()
    print(
        f"[+] Phát hiện GPU vật lý: {gpu_info or 'Không đọc được qua nvidia-smi, vẫn tiến hành cài bản CUDA'}"
    )
    print(
        "[*] Đang cài đặt PyTorch + Torchvision bản NVIDIA CUDA (cu121) và transformers==4.49.0..."
    )
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "transformers==4.49.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu121",
    ]
    subprocess.run(cmd, check=True)
    print("[+] Hoàn tất cài đặt PyTorch CUDA! Hãy chạy lại `python app.py`.")


def classify_legal_doc_type(doc_number: str, title: str, text: str = "") -> str:
    code = (doc_number or "").upper().strip()
    head = f"{title} {text[:250]}".upper()

    if re.search(r"/(?:QH\d*|L-CTN|VBHN-VPQH)\b", code):
        return "Luật / Bộ luật"
    if re.search(r"/(?:NĐ-CP|ND-CP|CP)\b", code):
        return "Nghị định"
    if re.search(r"/(?:TT|TTLT)-[A-Z0-9\-]+\b", code) or "/TT-" in code:
        return "Thông tư"
    if re.search(r"/(?:QĐ|QD)-[A-Z0-9\-]+\b", code) or "/QĐ-" in code:
        if "QUY CHẾ" in head or "QUY ĐỊNH" in head:
            return "Quyết định (Quy chế / Quy định)"
        return "Quyết định"
    if re.search(r"/(?:NQ|CT)-[A-Z0-9\-]+\b", code):
        return "Nghị quyết / Chỉ thị"

    if re.search(r"\b(BỘ LUẬT|LUẬT)\b", head):
        return "Luật / Bộ luật"
    if "NGHỊ ĐỊNH" in head or "NGHI DINH" in head:
        return "Nghị định"
    if "THÔNG TƯ" in head or "THONG TU" in head:
        return "Thông tư"
    if "QUY CHẾ" in head or "QUY ĐỊNH" in head or "HƯỚNG DẪN" in head:
        return "Quy chế / Quy định"
    if "QUYẾT ĐỊNH" in head or "QUYET DINH" in head:
        return "Quyết định"
    if "NGHỊ QUYẾT" in head or "CHỈ THỊ" in head:
        return "Nghị quyết / Chỉ thị"

    return "Văn bản pháp quy khác"


def tokenize(text: str, use_word_segmenter: bool = True) -> list[str]:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return []

    base_tokens = TOKEN_RE.findall(cleaned)
    if not use_word_segmenter:
        return base_tokens

    if HAS_PYVI and ViTokenizer is not None:
        try:
            segmented = ViTokenizer.tokenize(cleaned)
            seg_tokens = TOKEN_RE.findall(segmented)
            compound_only = [tok for tok in seg_tokens if "_" in tok]
            return base_tokens + compound_only
        except Exception:
            return base_tokens

    extra_compounds: list[str] = []
    for phrase in FALLBACK_COMPOUND_TERMS:
        if phrase in cleaned:
            extra_compounds.append(phrase.replace(" ", "_"))
    return base_tokens + extra_compounds


def _extract_doc_number_hint(title: str, text: str) -> str:
    combined = f"{title} {text[:500]}"
    normative_pattern = re.compile(
        r"\b(\d{1,4}/(?:\d{4}/)?(?:TT-BCA|NĐ-CP|ND-CP|QH\d{1,2}|VBHN-VPQH|QĐ-BCA|QD-BCA|QĐ-T01|QD-T01|QĐ-HVANND|TTLT-[A-Z0-9\-]+|CP|L-CTN|NQ-CP|TT-[A-Z0-9\-]+|QĐ-[A-Z0-9\-]+))\b",
        re.IGNORECASE,
    )
    match = normative_pattern.search(combined)
    if match:
        return match.group(1).upper()
    return "CHƯA_RÕ_SỐ_HIỆU"


def load_corpus(path: Path, active_only: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {path}")

    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    required = {"id", "title", "text", "source"}
    if not rows or any(required - row.keys() for row in rows):
        raise ValueError("Corpus rỗng hoặc thiếu id/title/text/source")

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["id"] = str(item["id"])
        item.setdefault("status", "active")
        doc_num = item.get("doc_number") or _extract_doc_number_hint(
            item["title"], item["text"]
        )
        item["doc_number"] = str(doc_num)
        item["doc_type"] = item.get("doc_type") or classify_legal_doc_type(
            item["doc_number"], item["title"], item["text"]
        )
        if active_only and item.get("status") != "active":
            continue
        normalized_rows.append(item)

    return normalized_rows


def save_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    required = {"qid", "question", "relevant_ids"}
    if not rows or any(required - row.keys() for row in rows):
        raise ValueError("Queries rỗng hoặc thiếu qid/question/relevant_ids")
    return rows


def check_ollama_status(
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    resolved_url = (
        base_url or os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    ).rstrip("/")
    resolved_model = model_name or os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
    root_url = re.sub(r"/v1$", "", resolved_url)

    try:
        resp = requests.get(f"{root_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        model_ready = any(resolved_model.lower() in m.lower() for m in models)
        return {
            "online": True,
            "model_ready": model_ready,
            "configured_model": resolved_model,
            "available_models": models,
            "base_url": resolved_url,
            "message": (
                f"Ollama đang hoạt động ({resolved_model} sẵn sàng)"
                if model_ready
                else f"Ollama đang chạy nhưng chưa thấy model '{resolved_model}'."
            ),
        }
    except Exception as exc:
        return {
            "online": False,
            "model_ready": False,
            "configured_model": resolved_model,
            "available_models": [],
            "base_url": resolved_url,
            "message": f"Chế độ Offline ({exc.__class__.__name__})",
        }


def normalize_legal_query_heuristic(
    question: str,
    corpus_doc_numbers: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Tầng 1 của Mô-đun Chuẩn hóa Câu hỏi Pháp lý (chạy tức thì < 1ms, không phụ thuộc LLM):
    1. Giải nghĩa các từ viết tắt pháp lý phổ biến (`vphc` -> `vi phạm hành chính`, `gplx` -> `giấy phép lái xe`,
       `anm` -> `an ninh mạng`, `t01` -> `Học viện An ninh nhân dân`...).
    2. Ánh xạ số hiệu viết tắt trong câu hỏi (VD: `nghị định 168`, `nd 168`, `luật 116`, `thông tư 137`)
       sang đúng Số hiệu đầy đủ của văn bản vừa được nạp trong kho (`168/2024/NĐ-CP`, `116/2025/QH15`...).
    3. Bổ sung thuật ngữ pháp điển tương đương cho các cụm từ đời thường (VD: `uống rượu bia` ->
       `trong máu hoặc hơi thở có nồng độ cồn`, `vượt đèn đỏ` -> `không chấp hành hiệu lệnh của đèn tín hiệu giao thông`).
    """
    normalized = (question or "").strip()
    if not normalized:
        return ("", [])

    applied_notes: list[str] = []

    # 1. Thay thế từ viết tắt pháp lý
    for pat, full_term in LEGAL_ABBREVIATION_MAP:
        if pat.search(normalized):
            normalized = pat.sub(full_term, normalized)
            applied_notes.append(f"Mở rộng viết tắt -> '{full_term}'")

    # 2. Đối chiếu tự động với danh sách Số hiệu văn bản hiện có trong kho (Corpus-Aware Mapping)
    if corpus_doc_numbers:
        short_num_map: dict[str, str] = {}
        for full_code in corpus_doc_numbers:
            if full_code and full_code != "CHƯA_RÕ_SỐ_HIỆU":
                m = re.match(r"^(\d{1,4})/", full_code)
                if m:
                    short_num_map[m.group(1)] = full_code

        # Tìm các cụm như "nghị định 168", "nđ 168", "nd 168", "thông tư 137", "tt 137", "luật 116", "quyết định 15"
        def _expand_short_doc_ref(match: re.Match[str]) -> str:
            prefix_word = match.group(1)
            short_no = match.group(2)
            if short_no in short_num_map and short_num_map[short_no] not in normalized.upper():
                full_code = short_num_map[short_no]
                applied_notes.append(f"Khớp số hiệu kho luật -> '{full_code}'")
                return f"{prefix_word} số {full_code}"
            return match.group(0)

        normalized = re.sub(
            r"\b(nghị\s+định|nđ|nd|thông\s+tư|tt|luật|quyết\s+định|qđ|qd)\s+(?:số\s+)?(\d{1,4})(?!\s*/)\b",
            _expand_short_doc_ref,
            normalized,
            flags=re.IGNORECASE,
        )

    # 3. Bổ sung thuật ngữ pháp điển cho ngôn ngữ đời thường
    added_legal_terms: list[str] = []
    for pat, legal_phrase in COLLOQUIAL_LEGAL_EXPANSIONS:
        if pat.search(normalized) and legal_phrase.lower() not in normalized.lower():
            added_legal_terms.append(legal_phrase)
            applied_notes.append(f"Chuẩn hóa thuật ngữ -> '{legal_phrase}'")

    if added_legal_terms:
        normalized = f"{normalized} ({', '.join(added_legal_terms)})"

    return (normalized, applied_notes)


def rewrite_query_with_trace(
    history: str,
    question: str,
    *,
    timeout: int | float = DEFAULT_LLM_TIMEOUT,
    log_path: Path | None = None,
    use_ollama_default: bool = True,
    corpus_doc_numbers: list[str] | None = None,
    corpus_catalog_summary: str = "",
) -> dict[str, Any]:
    """
    Mô-đun Chuẩn hóa Câu hỏi Pháp lý & Viết lại Hội thoại (2 tầng kết hợp):
    - Tầng 1 (Heuristic & Corpus-Aware): Chuẩn hóa ngay lập tức từ viết tắt, khẩu ngữ đời thường
      và ánh xạ số hiệu văn bản mới trong kho (kể cả ở lượt hỏi đầu tiên khi `history` rỗng!).
    - Tầng 2 (LLM Local `qwen2.5:7b` với `temperature=0`):
      + Nếu có `history` (hội thoại nhiều lượt) HOẶC câu hỏi cần chuẩn hóa sâu theo danh mục luật trong kho,
        gọi LLM để viết lại thành câu hỏi pháp lý độc lập, rõ nghĩa.
      + Nếu LLM không bật hoặc quá thời gian chờ, hệ thống dùng kết quả đã chuẩn hóa ở Tầng 1!
    """
    original = (question or "").strip()
    clean_history = (history or "").strip()
    started = time.perf_counter()

    # Tầng 1: Chuẩn hóa thuật ngữ pháp lý & khớp số hiệu văn bản trong kho
    heuristic_query, heuristic_notes = normalize_legal_query_heuristic(
        original, corpus_doc_numbers=corpus_doc_numbers
    )

    # Kiểm tra xem câu hỏi có đại từ tham chiếu hội thoại hoặc cần LLM viết lại không
    has_pronoun_ref = bool(
        re.search(
            r"\b(nó|thủ\s+tục\s+này|quy\s+định\s+này|quy\s+định\s+đó|trường\s+hợp\s+này|trường\s+hợp\s+đó|mức\s+phạt\s+trên|hành\s+vi\s+trên|điều\s+này|văn\s+bản\s+này|như\s+thế\s+nào\s*\?$)\b",
            original.lower(),
        )
    )

    # Nếu không có lịch sử hội thoại VÀ Tầng 1 đã chuẩn hóa xong (hoặc câu hỏi đã rất chuẩn xác)
    # nhưng vẫn ưu tiên gọi LLM nếu người dùng bật `use_ollama_default` để chuẩn hóa câu hỏi khẩu ngữ
    if not clean_history and not use_ollama_default:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        trace = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "original": original,
            "rewritten": heuristic_query,
            "history": "",
            "used_rewrite": heuristic_query != original,
            "status": "normalized_rule" if heuristic_query != original else "no_history",
            "notes": heuristic_notes,
            "model": "Rule-Based Legal Normalizer",
            "latency_ms": latency_ms,
            "error": None,
        }
        _append_rewrite_log(trace, log_path)
        return trace

    try:
        if use_ollama_default:
            base_url = os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/")
            api_key = os.environ.get("LLM_API_KEY", DEFAULT_LLM_API_KEY)
            model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
        else:
            base_url = os.environ.get(
                "LLM_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/")
            api_key = os.environ["LLM_API_KEY"]
            model = os.environ["LLM_MODEL"]

        catalog_ctx = (
            f"\nCác văn bản pháp luật đang có trong kho dữ liệu:\n{corpus_catalog_summary}\n"
            if corpus_catalog_summary
            else ""
        )

        if clean_history:
            system_prompt = f"""
Bạn là chuyên gia chuẩn hóa câu hỏi tra cứu pháp luật Việt Nam.
{catalog_ctx}
Yêu cầu bắt buộc:
- Dùng lịch sử hội thoại để thay thế các đại từ hoặc tham chiếu mơ hồ (như "nó", "thủ tục này", "quy định đó", "mức phạt trên") bằng đúng chủ đề/văn bản đã nhắc ở lượt trước.
- Chuyển đổi các từ viết tắt hoặc khẩu ngữ đời thường sang thuật ngữ pháp lý chuẩn (ví dụ: "uống rượu bia" -> "trong máu hoặc hơi thở có nồng độ cồn", "bằng lái" -> "giấy phép lái xe").
- Tuyệt đối không tự bịa thêm tên người, tổ chức, thời gian hoặc điều kiện mới ngoài câu hỏi.
- Chỉ trả về duy nhất 1 câu hỏi tiếng Việt đã chuẩn hóa trên 1 dòng, không giải thích.
""".strip()
            user_content = (
                f"Lịch sử hội thoại:\n{clean_history}\n\n"
                f"Câu hỏi mới:\n{heuristic_query}"
            )
        else:
            system_prompt = f"""
Bạn là bộ chuẩn hóa câu hỏi tra cứu văn bản pháp luật Việt Nam.
{catalog_ctx}
Yêu cầu bắt buộc:
- Chuyển câu hỏi đời thường, khẩu ngữ hoặc viết tắt của người dùng thành câu hỏi sử dụng thuật ngữ pháp lý chính quy trong văn bản luật Việt Nam (ví dụ: "đi xe đạp bấm điện thoại phạt bao nhiêu" -> "Mức xử phạt vi phạm hành chính đối với người điều khiển xe đạp sử dụng điện thoại").
- Giữ nguyên mọi số hiệu văn bản, con số và chủ thể mà người dùng đã nêu; tuyệt đối không tự bịa thêm dữ kiện mới.
- Chỉ trả về duy nhất 1 câu hỏi đã chuẩn hóa trên 1 dòng, không giải thích, không thêm dấu ngoặc kép.
""".strip()
            user_content = f"Câu hỏi cần chuẩn hóa:\n{heuristic_query}"

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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()

        raw_rewritten = (
            response.json()["choices"][0]["message"]["content"].strip()
        )
        raw_rewritten = re.sub(
            r"<think>.*?</think>", "", raw_rewritten, flags=re.DOTALL
        ).strip()
        rewritten = (
            raw_rewritten.splitlines()[0].strip().strip('"').strip("'")
            if raw_rewritten
            else heuristic_query
        )
        # Bỏ tiền tố thừa nếu LLM lỡ in "Câu hỏi chuẩn hóa: ..."
        rewritten = re.sub(
            r"^(?:Câu\s+hỏi\s+(?:đã\s+)?(?:chuẩn\s+hóa|viết\s+lại)\s*[\:\-]\s*)",
            "",
            rewritten,
            flags=re.IGNORECASE,
        ).strip()

        final_query = rewritten or heuristic_query
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        trace = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "original": original,
            "rewritten": final_query,
            "history": clean_history,
            "used_rewrite": final_query != original,
            "status": (
                "rewritten_llm"
                if final_query != original
                else ("normalized_rule" if heuristic_notes else "unchanged")
            ),
            "notes": heuristic_notes,
            "model": model,
            "latency_ms": latency_ms,
            "error": None,
        }
        _append_rewrite_log(trace, log_path)
        return trace

    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        trace = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "original": original,
            "rewritten": heuristic_query,
            "history": clean_history,
            "used_rewrite": heuristic_query != original,
            "status": (
                "normalized_rule_fallback"
                if heuristic_query != original
                else "fallback_original"
            ),
            "notes": heuristic_notes,
            "model": "Rule-Based Fallback",
            "latency_ms": latency_ms,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        _append_rewrite_log(trace, log_path)
        return trace


def rewrite_query(history: str, question: str) -> str:
    """Hàm tương thích ngược 100% với mã nguồn gốc `hybrid_search.py` và `--self-check`."""
    original = question.strip()
    if not history.strip():
        return original

    has_explicit_env = "LLM_API_KEY" in os.environ and "LLM_MODEL" in os.environ
    ollama_enabled = os.environ.get("USE_LOCAL_OLLAMA", "0") == "1"
    if not has_explicit_env and not ollama_enabled:
        return original

    trace = rewrite_query_with_trace(
        history,
        question,
        timeout=int(os.environ.get("LLM_TIMEOUT", "30")),
        use_ollama_default=ollama_enabled,
    )
    return str(trace["rewritten"])


def _append_rewrite_log(trace: dict[str, Any], log_path: Path | None = None) -> None:
    target = log_path or (get_app_base_dir() / "logs" / "query_rewrite_audit.jsonl")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
    except OSError:
        pass


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


def _compute_row_fingerprint(row: dict[str, Any]) -> str:
    raw = f"{row.get('id', '')}||{row.get('title', '')}||{row.get('text', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_for_dedup(text: str) -> str:
    cleaned = re.sub(r"\W+", " ", (text or "").lower(), flags=re.UNICODE).strip()
    return re.sub(r"\s+", " ", cleaned)


def _compute_content_hash(text: str) -> str:
    norm = _normalize_for_dedup(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class HybridRetriever:
    """Bộ truy xuất tìm kiếm lai (BM25 + Dense Retrieval BAAI/bge-m3 + RRF)."""

    def __init__(
        self,
        corpus_path: Path,
        model_name: str = MODEL_NAME,
        embeddings_path: Path | None = None,
        use_word_segmenter: bool = True,
    ):
        self.corpus_path = Path(corpus_path).resolve()
        self.model_name = model_name
        self.use_word_segmenter = use_word_segmenter
        self.embeddings_path = (
            Path(embeddings_path).resolve()
            if embeddings_path
            else self.corpus_path.with_name("embeddings.npy")
        )
        self.meta_path = self.embeddings_path.with_name("embeddings_meta.json")
        self.bm25_cache_path = self.embeddings_path.with_name("bm25_tokens_cache.json")

        self.hardware_info = detect_hardware_accelerator()
        if self.hardware_info.get("cuda_mismatch_warning"):
            print(f"\n[!] CẢNH BÁO GPU: {self.hardware_info['cuda_mismatch_warning']}\n")

        self.batch_size = int(self.hardware_info.get("batch_size", 8))

        local_model_dir = get_app_base_dir() / "models" / model_name.replace("/", "_")
        resolved_model_path = (
            str(local_model_dir) if local_model_dir.exists() else model_name
        )

        target_device = self.hardware_info["device"]
        st_device = target_device if target_device in ("cuda", "mps", "cpu") else "cpu"

        t0 = time.perf_counter()
        print(f"[1/2] Đang nạp mô hình {model_name} trên thiết bị [{st_device.upper()}]...")
        try:
            self.model = SentenceTransformer(
                resolved_model_path,
                device=st_device,
                local_files_only=True,
            )
        except Exception:
            print(
                "      (Chưa thấy cache offline đầy đủ, đang kết nối HuggingFace để tải lần đầu...)"
            )
            self.model = SentenceTransformer(
                resolved_model_path,
                device=st_device,
                local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
            )

        if st_device == "cuda" and self.hardware_info.get("use_fp16"):
            try:
                self.model.half()
            except Exception:
                pass

        self.model.max_seq_length = MODEL_MAX_SEQ_LENGTH
        print(
            f"      -> Đã nạp xong mô hình sau {round((time.perf_counter() - t0)*1000)} ms."
        )

        self.rows: list[dict[str, Any]] = []
        self.texts: list[str] = []
        self.bm25_tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray = np.empty((0, 1024), dtype=np.float32)
        self.corpus_token_lengths: list[int] = []
        self.last_index_stats: dict[str, Any] = {}

        self._cached_encode_query = lru_cache(maxsize=256)(self._encode_query_raw)
        self.reload_or_sync_index()

    def _encode_query_raw(self, query: str) -> np.ndarray:
        encode_query = getattr(self.model, "encode_query", self.model.encode)
        vec = encode_query(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vec, dtype=np.float32)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        encode_documents = getattr(
            self.model, "encode_document", self.model.encode
        )
        vecs = encode_documents(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True if len(texts) > 20 else False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def get_active_doc_numbers(self) -> list[str]:
        """Lấy danh sách các Số hiệu văn bản đang có hiệu lực trong kho."""
        return sorted(
            {
                str(r.get("doc_number", ""))
                for r in self.rows
                if r.get("status", "active") == "active"
                and r.get("doc_number")
                and r.get("doc_number") != "CHƯA_RÕ_SỐ_HIỆU"
            }
        )

    def build_corpus_catalog_hint(self, max_docs: int = 15) -> str:
        """
        Tạo bản tóm tắt danh mục các văn bản pháp luật đang có trong kho (đặc biệt là các
        văn bản mới nạp) để cung cấp ngữ cảnh cho LLM khi chuẩn hóa câu hỏi!
        """
        doc_map: dict[str, str] = {}
        for r in reversed(self.rows):
            if r.get("status", "active") != "active":
                continue
            dnum = str(r.get("doc_number", "CHƯA_RÕ_SỐ_HIỆU"))
            if dnum == "CHƯA_RÕ_SỐ_HIỆU" or dnum in doc_map:
                continue
            dtype = str(r.get("doc_type", "Văn bản"))
            title_head = str(r.get("title", "")).split(" - Điều")[0].strip()
            doc_map[dnum] = f"- {dtype} {dnum}: {title_head[:110]}"
            if len(doc_map) >= max_docs:
                break
        return "\n".join(doc_map.values())

    def generate_suggested_queries(self, max_suggestions: int = 6) -> list[dict[str, str]]:
        """
        Tự động sinh bộ câu hỏi gợi ý chuẩn hóa từ chính các văn bản trong kho
        (bỏ phần số hiệu văn bản dài dòng và bỏ qua các điều khoản thủ tục cuối văn bản như
        Trách nhiệm thi hành, Quy định chuyển tiếp, Hiệu lực thi hành).
        """
        suggestions: list[dict[str, str]] = []
        seen_docs: set[str] = set()
        skip_boilerplate_re = re.compile(
            r"^(trách\s+nhiệm\s+thi\s+hành|hiệu\s+lực\s+thi\s+hành|quy\s+định\s+chuyển\s+tiếp|"
            r"điều\s+khoản\s+thi\s+hành|tổ\s+chức\s+thực\s+hiện|phạm\s+vi\s+điều\s+chỉnh|"
            r"đối\s+tượng\s+áp\s+dụng|giải\s+thích\s+từ\s+ngữ)\b",
            re.IGNORECASE,
        )

        # Duyệt các văn bản mới nạp nhưng ưu tiên các điều khoản nội dung thực chất
        doc_candidates: dict[str, list[str]] = {}
        for r in reversed(self.rows):
            if r.get("status", "active") != "active":
                continue
            dnum = str(r.get("doc_number", "CHƯA_RÕ_SỐ_HIỆU"))
            if dnum == "CHƯA_RÕ_SỐ_HIỆU":
                continue
            title = str(r.get("title", ""))
            m_art = re.search(r"(Điều\s+\d+[\.\:]\s*[^\(\n]{6,95})", title)
            if m_art:
                art_part = m_art.group(1).strip()
                topic = re.sub(r"^Điều\s+\d+[\.\:]\s*", "", art_part).strip()
                # Loại bỏ mọi cụm số hiệu nằm trong ngoặc vuông hoặc cuối câu nếu có
                topic = re.sub(r"\[[^\]]+\]", "", topic).strip(" .-–—:;,")
                if len(topic) >= 10 and not skip_boilerplate_re.search(topic):
                    doc_candidates.setdefault(dnum, []).append(topic)

        for dnum, topics in doc_candidates.items():
            if dnum in seen_docs:
                continue
            # Chọn điều khoản nội dung thực chất (ưu tiên điều khoản ở phần thân văn bản)
            chosen_topic = topics[len(topics) // 2] if len(topics) > 2 else (topics[-1] if topics else "")
            chosen_topic = re.sub(r"\([^\)]*\d+\s*/[^\)]*\)", "", chosen_topic).strip(" .-–—:;,")
            if chosen_topic:
                seen_docs.add(dnum)
                short_label = chosen_topic[:48] + ("..." if len(chosen_topic) > 48 else "")
                suggestions.append(
                    {
                        "label": short_label,
                        "query": f"Quy định về {chosen_topic.lower()} như thế nào?",
                        "history": "",
                    }
                )
            if len(suggestions) >= 3:
                break

        default_samples = [
            {
                "label": "Chế độ tiền lương & phụ cấp",
                "query": "Phó Tổng Giám đốc Ngân hàng Chính sách xã hội được xếp lương theo bảng lương như thế nào?",
                "history": "",
            },
            {
                "label": "Thử chuẩn hóa khẩu ngữ ('uống rượu bia đi xe đạp')",
                "query": "Người đi xe đạp uống rượu bia hoặc vừa đi vừa bấm điện thoại bị phạt bao nhiêu tiền?",
                "history": "",
            },
            {
                "label": "Hỏi nối tiếp theo ngữ cảnh ('Nó được quy định...')",
                "query": "Nó được quy định cụ thể tại điều khoản nào?",
                "history": "Người dùng: Chế độ tiền lương của Phó Tổng Giám đốc Ngân hàng Chính sách xã hội",
            },
            {
                "label": "Thời hiệu xử phạt VPHC",
                "query": "Thời hiệu xử phạt vphc trong lĩnh vực xuất bản điện tử chưa đăng ký?",
                "history": "",
            },
        ]
        for s in default_samples:
            if len(suggestions) < max_suggestions:
                suggestions.append(s)
        return suggestions

    def reload_or_sync_index(self) -> dict[str, Any]:
        started = time.perf_counter()
        print("[2/2] Đang đồng bộ hóa chỉ mục BM25 và Vector Embeddings...")
        self.rows = load_corpus(self.corpus_path, active_only=False)
        self.texts = [f"{row['title']}. {row['text']}" for row in self.rows]
        current_fps = [_compute_row_fingerprint(row) for row in self.rows]
        segmenter_name = (
            "pyvi"
            if (HAS_PYVI and self.use_word_segmenter)
            else "regex_plus_compounds"
        )

        bm25_token_map: dict[str, list[str]] = {}
        tok_len_map: dict[str, int] = {}
        if self.bm25_cache_path.exists():
            try:
                b_cache = json.loads(
                    self.bm25_cache_path.read_text(encoding="utf-8")
                )
                if b_cache.get("segmenter") == segmenter_name:
                    bm25_token_map = b_cache.get("tokens_by_fp", {})
                    tok_len_map = b_cache.get("lengths_by_fp", {})
            except Exception:
                bm25_token_map = {}
                tok_len_map = {}

        missing_bm25_indices = [
            idx
            for idx, fp in enumerate(current_fps)
            if fp not in bm25_token_map or fp not in tok_len_map
        ]

        if missing_bm25_indices:
            missing_bm25_texts = [self.texts[i] for i in missing_bm25_indices]
            tokenized_batch = self.model.tokenizer(
                missing_bm25_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_length=True,
            )
            batch_lengths = list(map(int, tokenized_batch["length"]))
            for idx, tok_len in zip(missing_bm25_indices, batch_lengths):
                fp = current_fps[idx]
                bm25_token_map[fp] = tokenize(
                    self.texts[idx], self.use_word_segmenter
                )
                tok_len_map[fp] = tok_len

        self.bm25_tokenized_corpus = [bm25_token_map[fp] for fp in current_fps]
        self.corpus_token_lengths = [int(tok_len_map[fp]) for fp in current_fps]
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)

        if missing_bm25_indices:
            self._save_bm25_cache(current_fps, segmenter_name)

        cached_map: dict[str, np.ndarray] = {}
        reused_count = 0
        encoded_count = 0

        if self.embeddings_path.exists() and self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if meta.get("model_name") == self.model_name:
                    saved_matrix = np.load(self.embeddings_path)
                    saved_fps: list[str] = meta.get("fingerprints", [])
                    if len(saved_fps) == len(saved_matrix):
                        for fp, vec in zip(saved_fps, saved_matrix):
                            cached_map[fp] = vec
            except Exception:
                cached_map = {}

        missing_indices = [
            idx for idx, fp in enumerate(current_fps) if fp not in cached_map
        ]

        if not missing_indices and len(current_fps) > 0:
            self.embeddings = np.vstack(
                [cached_map[fp] for fp in current_fps]
            ).astype(np.float32)
            reused_count = len(current_fps)
        else:
            missing_texts = [self.texts[idx] for idx in missing_indices]
            print(
                f"      -> Đang tính Embedding cho {len(missing_texts)} điều khoản mới trên [{self.hardware_info['device_name']}]..."
            )
            new_vecs = self._encode_texts(missing_texts)
            for idx, vec in zip(missing_indices, new_vecs):
                cached_map[current_fps[idx]] = vec
            self.embeddings = np.vstack(
                [cached_map[fp] for fp in current_fps]
            ).astype(np.float32)
            reused_count = len(current_fps) - len(missing_indices)
            encoded_count = len(missing_indices)
            self._save_embeddings_cache(current_fps)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.last_index_stats = {
            "total_documents": len(self.rows),
            "active_documents": sum(
                1 for r in self.rows if r.get("status", "active") == "active"
            ),
            "replaced_documents": sum(
                1 for r in self.rows if r.get("status", "active") != "active"
            ),
            "reused_embeddings": reused_count,
            "newly_encoded_embeddings": encoded_count,
            "sync_time_ms": elapsed_ms,
            "word_segmenter": segmenter_name,
            "hardware": self.hardware_info,
        }
        return self.last_index_stats

    def _save_bm25_cache(
        self, fingerprints: list[str] | None = None, segmenter_name: str | None = None
    ) -> None:
        fps = fingerprints or [_compute_row_fingerprint(row) for row in self.rows]
        seg = segmenter_name or (
            "pyvi" if (HAS_PYVI and self.use_word_segmenter) else "regex_plus_compounds"
        )
        try:
            tokens_by_fp = {
                fp: toks for fp, toks in zip(fps, self.bm25_tokenized_corpus)
            }
            lengths_by_fp = {
                fp: ln for fp, ln in zip(fps, self.corpus_token_lengths)
            }
            payload = {
                "segmenter": seg,
                "count": len(fps),
                "tokens_by_fp": tokens_by_fp,
                "lengths_by_fp": lengths_by_fp,
            }
            self.bm25_cache_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _save_embeddings_cache(self, fingerprints: list[str] | None = None) -> None:
        fps = fingerprints or [_compute_row_fingerprint(row) for row in self.rows]
        try:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.embeddings_path, self.embeddings)
            meta = {
                "model_name": self.model_name,
                "dimension": int(self.embeddings.shape[1]) if self.embeddings.size else 1024,
                "count": len(self.rows),
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "fingerprints": fps,
            }
            self.meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def update_documents_status(
        self,
        *,
        chunk_id: str | None = None,
        doc_number: str | None = None,
        scope: str = "single",
        new_status: str = "replaced",
        replaced_by: str = "",
    ) -> dict[str, Any]:
        """
        Đánh dấu hết hiệu lực (hoặc khôi phục hiệu lực) cho:
        - Từng điều khoản riêng lẻ (`scope="single"`, dựa theo `chunk_id`)
        - Hoặc toàn bộ các điều khoản thuộc cùng một văn bản pháp luật (`scope="document"`, dựa theo `doc_number`)
        và lưu trực tiếp vào cơ sở dữ liệu `corpus.jsonl`.
        """
        norm_scope = (scope or "single").strip().lower()
        norm_status = "active" if (new_status or "").strip().lower() == "active" else "replaced"
        target_id = (chunk_id or "").strip()
        target_doc = (doc_number or "").strip().upper()
        note_str = (replaced_by or "").strip() or ("Hết hiệu lực" if norm_status == "replaced" else "")
        now_iso = dt.datetime.now().isoformat(timespec="seconds")

        updated_count = 0
        for row in self.rows:
            matched = False
            if norm_scope == "document":
                if not target_doc or target_doc == "CHƯA_RÕ_SỐ_HIỆU":
                    raise ValueError("Văn bản này chưa có Số hiệu xác định để đánh dấu toàn bộ văn bản.")
                if str(row.get("doc_number", "")).strip().upper() == target_doc:
                    matched = True
            else:
                if not target_id:
                    raise ValueError("Thiếu mã điều khoản (chunk_id) cần cập nhật trạng thái.")
                if str(row.get("id", "")).strip() == target_id:
                    matched = True

            if matched:
                row["status"] = norm_status
                if norm_status == "replaced":
                    row["replaced_by"] = note_str
                    row["replaced_at"] = now_iso
                else:
                    row.pop("replaced_by", None)
                    row.pop("replaced_at", None)
                updated_count += 1

        if updated_count == 0:
            raise ValueError("Không tìm thấy điều khoản hoặc văn bản phù hợp để cập nhật trạng thái hiệu lực.")

        save_corpus(self.corpus_path, self.rows)
        active_cnt = sum(1 for r in self.rows if r.get("status", "active") == "active")
        replaced_cnt = len(self.rows) - active_cnt
        if hasattr(self, "last_index_stats") and isinstance(self.last_index_stats, dict):
            self.last_index_stats["active_documents"] = active_cnt
            self.last_index_stats["replaced_documents"] = replaced_cnt

        return {
            "updated_count": updated_count,
            "scope": norm_scope,
            "new_status": norm_status,
            "replaced_by": note_str,
            "active_chunks": active_cnt,
            "replaced_chunks": replaced_cnt,
            "total_chunks": len(self.rows),
        }

    def delete_documents_by_doc_number(self, doc_number: str) -> dict[str, Any]:
        target = (doc_number or "").strip().upper()
        if not target:
            raise ValueError("Vui lòng cung cấp Số hiệu văn bản cần xóa.")

        keep_indices = [
            idx
            for idx, r in enumerate(self.rows)
            if str(r.get("doc_number", "")).strip().upper() != target
        ]
        removed_count = len(self.rows) - len(keep_indices)
        if removed_count == 0:
            raise ValueError(
                f"Không tìm thấy văn bản nào có số hiệu '{doc_number}' trong kho."
            )
        if not keep_indices:
            raise ValueError(
                "Không thể xóa toàn bộ kho dữ liệu (cần giữ lại ít nhất 1 văn bản)."
            )

        self.rows = [self.rows[i] for i in keep_indices]
        self.texts = [self.texts[i] for i in keep_indices]
        self.bm25_tokenized_corpus = [
            self.bm25_tokenized_corpus[i] for i in keep_indices
        ]
        self.corpus_token_lengths = [
            self.corpus_token_lengths[i] for i in keep_indices
        ]
        self.embeddings = self.embeddings[keep_indices]
        self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
        self._cached_encode_query.cache_clear()

        fps = [_compute_row_fingerprint(row) for row in self.rows]
        save_corpus(self.corpus_path, self.rows)
        self._save_embeddings_cache(fps)
        self._save_bm25_cache(fps)

        return {
            "removed_chunks": removed_count,
            "remaining_chunks": len(self.rows),
            "doc_number": doc_number,
        }

    def check_duplicate_document(
        self,
        new_chunks: list[dict[str, Any]],
        doc_number: str,
        doc_title: str = "",
    ) -> dict[str, Any]:
        clean_num = (doc_number or "").strip().upper()
        active_rows = [
            r for r in self.rows if r.get("status", "active") == "active"
        ]

        same_doc_num_rows = [
            r
            for r in active_rows
            if clean_num
            and clean_num not in ("CHƯA_RÕ_SỐ_HIỆU", "VB-T01-MOI", "VB-MOI")
            and str(r.get("doc_number", "")).strip().upper() == clean_num
        ]

        existing_hashes: dict[str, dict[str, Any]] = {
            _compute_content_hash(r.get("text", "")): r for r in active_rows
        }
        exact_duplicate_chunks: list[dict[str, Any]] = []
        new_unique_chunks_count = 0

        for ch in new_chunks:
            ch_hash = _compute_content_hash(ch.get("text", ""))
            if ch_hash in existing_hashes:
                matched_row = existing_hashes[ch_hash]
                exact_duplicate_chunks.append(
                    {
                        "new_id": ch.get("id"),
                        "new_title": ch.get("title"),
                        "existing_id": matched_row.get("id"),
                        "existing_doc_number": matched_row.get("doc_number"),
                        "existing_title": matched_row.get("title"),
                    }
                )
            else:
                new_unique_chunks_count += 1

        title_tokens = {
            t
            for t in TOKEN_RE.findall((doc_title or "").lower())
            if len(t) > 2 and not t.isdigit()
        }
        similar_doc_numbers: set[str] = set()
        if len(title_tokens) >= 4:
            for r in active_rows:
                r_doc = str(r.get("doc_number", "CHƯA_RÕ_SỐ_HIỆU"))
                if r_doc in ("CHƯA_RÕ_SỐ_HIỆU", clean_num):
                    continue
                r_tokens = set(TOKEN_RE.findall(str(r.get("title", "")).lower()))
                overlap = len(title_tokens.intersection(r_tokens)) / max(
                    1, len(title_tokens)
                )
                if overlap >= 0.65:
                    similar_doc_numbers.add(r_doc)

        has_same_doc = len(same_doc_num_rows) > 0
        has_dup_content = len(exact_duplicate_chunks) > 0
        has_conflict = has_same_doc or has_dup_content

        if has_same_doc and len(exact_duplicate_chunks) == len(new_chunks):
            conflict_level = "exact_duplicate"
            recommended_action = "replace_existing"
            warning_message = (
                f"CẢNH BÁO TRÙNG HOÀN TOÀN: Văn bản số hiệu '{clean_num}' đã tồn tại trong kho "
                f"({len(same_doc_num_rows)} điều khoản đang có hiệu lực) và toàn bộ {len(new_chunks)} điều khoản "
                f"tải lên đều trùng khớp nội dung! Hãy chọn 'Thay thế văn bản cũ' nếu muốn cập nhật lại."
            )
        elif has_same_doc:
            conflict_level = "same_doc_number"
            recommended_action = "replace_existing"
            warning_message = (
                f"CẢNH BÁO TRÙNG SỐ HIỆU: Kho dữ liệu đã có văn bản số hiệu '{clean_num}' "
                f"({len(same_doc_num_rows)} điều khoản hiện hành), trong đó có {len(exact_duplicate_chunks)}/{len(new_chunks)} "
                f"điều khoản trùng nội dung và {new_unique_chunks_count} điều khoản mới/sửa đổi. "
                f"Bạn muốn THAY THẾ toàn bộ văn bản '{clean_num}' cũ hay CHỈ BỔ SUNG {new_unique_chunks_count} điều khoản mới?"
            )
        elif has_dup_content:
            conflict_level = "partial_content_duplicate"
            recommended_action = "append_new_only"
            dup_docs = sorted(
                {str(d["existing_doc_number"]) for d in exact_duplicate_chunks}
            )
            warning_message = (
                f"CẢNH BÁO TRÙNG NỘI DUNG ĐIỀU KHOẢN: Phát hiện {len(exact_duplicate_chunks)}/{len(new_chunks)} "
                f"điều khoản tải lên bị trùng nội dung với văn bản đã có trong kho ({', '.join(dup_docs[:3])}). "
                f"Khuyến nghị chọn chế độ 'Chỉ bổ sung các điều khoản chưa trùng ({new_unique_chunks_count} điều)'."
            )
        else:
            conflict_level = "none"
            recommended_action = "add_all"
            warning_message = (
                f"Kiểm tra trùng lặp: Không phát hiện trùng số hiệu hay trùng nội dung điều khoản trong kho hiện tại "
                f"({len(new_chunks)} điều khoản mới hoàn toàn)."
            )

        return {
            "has_conflict": has_conflict,
            "conflict_level": conflict_level,
            "recommended_action": recommended_action,
            "warning_message": warning_message,
            "existing_same_doc_count": len(same_doc_num_rows),
            "duplicate_chunks_count": len(exact_duplicate_chunks),
            "new_unique_chunks_count": new_unique_chunks_count,
            "duplicate_details": exact_duplicate_chunks[:10],
            "similar_doc_numbers": sorted(similar_doc_numbers)[:5],
        }

    def add_or_replace_documents(
        self,
        new_rows: list[dict[str, Any]],
        *,
        replace_doc_number: str | None = None,
        replace_ids: list[str] | None = None,
        conflict_mode: str = "auto",
    ) -> dict[str, Any]:
        if not new_rows:
            raise ValueError("Danh sách điều khoản mới trống.")

        started = time.perf_counter()
        replaced_count = 0
        skipped_duplicates = 0

        new_doc_number = str(new_rows[0].get("doc_number") or "VĂN_BẢN_MỚI").strip()
        target_ids = set(map(str, replace_ids or []))
        target_doc_num = (replace_doc_number or "").strip().upper()

        if conflict_mode == "replace_existing" and not target_doc_num:
            if new_doc_number.upper() not in ("CHƯA_RÕ_SỐ_HIỆU", "VB-T01-MOI", "VB-MOI"):
                target_doc_num = new_doc_number.upper()

        if target_doc_num or target_ids:
            for row in self.rows:
                row_doc = str(row.get("doc_number", "")).strip().upper()
                if (target_doc_num and row_doc == target_doc_num) or (
                    str(row["id"]) in target_ids
                ):
                    if row.get("status", "active") == "active":
                        row["status"] = "replaced"
                        row["replaced_by"] = new_doc_number
                        row["replaced_at"] = dt.datetime.now().isoformat(
                            timespec="seconds"
                        )
                        replaced_count += 1

        active_hashes = {
            _compute_content_hash(r.get("text", ""))
            for r in self.rows
            if r.get("status", "active") == "active"
        }

        existing_ids = {str(r["id"]) for r in self.rows}
        prepared_rows: list[dict[str, Any]] = []

        for idx, raw_row in enumerate(new_rows, start=1):
            row = dict(raw_row)
            text_val = str(row.get("text", "")).strip()
            if conflict_mode == "append_new_only":
                if _compute_content_hash(text_val) in active_hashes:
                    skipped_duplicates += 1
                    continue

            doc_id = str(row.get("id") or f"DOC_{int(time.time())}_{idx}")
            while doc_id in existing_ids:
                doc_id = f"{doc_id}_{idx}"
            existing_ids.add(doc_id)

            row["id"] = doc_id
            row["title"] = str(row.get("title", "")).strip()
            row["text"] = text_val
            row["source"] = str(row.get("source", "Bổ sung nội bộ T01")).strip()
            row["doc_number"] = str(
                row.get("doc_number")
                or _extract_doc_number_hint(row["title"], row["text"])
            )
            row["doc_type"] = str(
                row.get("doc_type")
                or classify_legal_doc_type(
                    row["doc_number"], row["title"], row["text"]
                )
            )
            row["status"] = row.get("status", "active")
            row["added_at"] = row.get(
                "added_at", dt.datetime.now().isoformat(timespec="seconds")
            )
            prepared_rows.append(row)

        if prepared_rows:
            new_texts = [f"{r['title']}. {r['text']}" for r in prepared_rows]
            new_embeddings = self._encode_texts(new_texts)
            new_bm25_tokens = [
                tokenize(text, self.use_word_segmenter) for text in new_texts
            ]

            self.rows.extend(prepared_rows)
            self.texts.extend(new_texts)
            self.bm25_tokenized_corpus.extend(new_bm25_tokens)
            if self.embeddings.size == 0:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings]).astype(
                    np.float32
                )

            self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
            new_tok = self.model.tokenizer(
                new_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_length=True,
            )
            self.corpus_token_lengths.extend(list(map(int, new_tok["length"])))
            self._cached_encode_query.cache_clear()

        fps = [_compute_row_fingerprint(row) for row in self.rows]
        save_corpus(self.corpus_path, self.rows)
        self._save_embeddings_cache(fps)
        self._save_bm25_cache(fps)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "added_chunks": len(prepared_rows),
            "skipped_duplicate_chunks": skipped_duplicates,
            "replaced_chunks": replaced_count,
            "total_chunks": len(self.rows),
            "active_chunks": sum(
                1 for r in self.rows if r.get("status", "active") == "active"
            ),
            "elapsed_ms": elapsed_ms,
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        include_replaced: bool = False,
        doc_type_filter: str = "all",
    ) -> dict[str, object]:
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn 0")
        assert self.bm25 is not None

        filtered_out_indices: set[int] = set()
        norm_type_filter = (doc_type_filter or "all").strip().lower()

        for idx, row in enumerate(self.rows):
            if not include_replaced and row.get("status", "active") != "active":
                filtered_out_indices.add(idx)
                continue
            if norm_type_filter not in ("all", ""):
                r_type = str(row.get("doc_type", "")).lower()
                if norm_type_filter not in r_type:
                    filtered_out_indices.add(idx)

        started = time.perf_counter()
        bm25_scores = np.asarray(
            self.bm25.get_scores(tokenize(query, self.use_word_segmenter)),
            dtype=float,
        )
        if filtered_out_indices:
            bm25_scores[list(filtered_out_indices)] = -1e9
        bm25_order = ranked_ids(bm25_scores)
        bm25_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        query_embedding = self._cached_encode_query(query)
        dense_scores = np.asarray(self.embeddings @ query_embedding, dtype=float)
        if filtered_out_indices:
            dense_scores[list(filtered_out_indices)] = -1e9
        dense_order = ranked_ids(dense_scores)
        dense_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        hybrid_order, hybrid_scores = rrf(
            [bm25_order, dense_order], [bm25_weight, dense_weight], rrf_k
        )
        if filtered_out_indices:
            hybrid_scores[list(filtered_out_indices)] = -1e9
            hybrid_order = [
                idx for idx in hybrid_order if idx not in filtered_out_indices
            ]
        fusion_ms = (time.perf_counter() - started) * 1000

        bm25_rank_map = {doc_idx: r for r, doc_idx in enumerate(bm25_order, start=1)}
        dense_rank_map = {doc_idx: r for r, doc_idx in enumerate(dense_order, start=1)}

        def pack(order: list[int], scores: np.ndarray) -> list[dict[str, object]]:
            valid_order = [idx for idx in order if idx not in filtered_out_indices]
            return [
                {
                    "rank": rank,
                    "id": self.rows[index]["id"],
                    "title": self.rows[index]["title"],
                    "text": self.rows[index]["text"],
                    "doc_number": self.rows[index].get("doc_number", "CHƯA_RÕ_SỐ_HIỆU"),
                    "doc_type": self.rows[index].get(
                        "doc_type",
                        classify_legal_doc_type(
                            str(self.rows[index].get("doc_number", "")),
                            self.rows[index]["title"],
                            self.rows[index]["text"],
                        ),
                    ),
                    "status": self.rows[index].get("status", "active"),
                    "replaced_by": self.rows[index].get("replaced_by"),
                    "score": round(float(scores[index]), 6),
                    "bm25_rank": bm25_rank_map.get(index),
                    "dense_rank": dense_rank_map.get(index),
                    "bm25_score": round(float(bm25_scores[index]), 4),
                    "dense_score": round(float(dense_scores[index]), 4),
                    "source": self.rows[index]["source"],
                }
                for rank, index in enumerate(valid_order[:top_k], start=1)
            ]

        return {
            "bm25": pack(bm25_order, bm25_scores),
            "dense": pack(dense_order, dense_scores),
            "hybrid": pack(hybrid_order, hybrid_scores),
            "latency_ms": {
                "bm25": round(bm25_ms, 2),
                "dense": round(dense_ms, 2),
                "fusion": round(fusion_ms, 2),
                "total": round(bm25_ms + dense_ms + fusion_ms, 2),
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
) -> dict[str, Any]:
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
        retriever._cached_encode_query.cache_clear()
        for query_row in queries:
            retriever.search(str(query_row["question"]), **search_options)

    for measured_pass in range(measured_passes):
        retriever._cached_encode_query.cache_clear()
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
    if hasattr(os, "sysconf_names") and {"SC_PAGE_SIZE", "SC_PHYS_PAGES"}.issubset(
        os.sysconf_names
    ):
        memory_gb = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3,
            2,
        )

    report = {
        "documents": len(retriever.rows),
        "queries": len(queries),
        "word_segmenter": retriever.last_index_stats.get("word_segmenter", "regex"),
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
            "hardware_accelerator": retriever.hardware_info,
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
                "p99": round(float(np.percentile(values, 99)), 2),
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
    return report


def chat(retriever: HybridRetriever, **search_options: object) -> None:
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

        trace = rewrite_query_with_trace(
            history,
            question,
            use_ollama_default=True,
            corpus_doc_numbers=retriever.get_active_doc_numbers(),
            corpus_catalog_summary=retriever.build_corpus_catalog_hint(),
        )
        rewritten = str(trace["rewritten"])
        result = retriever.search(rewritten, **search_options)
        print(
            json.dumps(
                {
                    "original": question,
                    "rewritten": rewritten,
                    "status": trace["status"],
                },
                ensure_ascii=False,
            )
        )
        print("Trợ lý: Tôi tìm thấy các tài liệu gần nhất:")
        for item in result["hybrid"]:
            print(
                f"- [{item['doc_type']}] {item['id']}: {item['title']} | {item['source']}"
            )
        print("Đây là kết quả truy xuất, không phải kết luận pháp lý.")
        history = f"{history}\nNgười dùng: {question}\nCâu hỏi chuẩn hóa: {rewritten}".strip()


def self_check() -> None:
    assert tokenize("Đăng ký doanh nghiệp!", use_word_segmenter=False) == [
        "đăng",
        "ký",
        "doanh",
        "nghiệp",
    ]
    seg_tokens = tokenize("Đăng ký doanh nghiệp!", use_word_segmenter=True)
    assert (
        "đăng" in seg_tokens
        and ("đăng_ký_doanh_nghiệp" in seg_tokens or "đăng_ký" in seg_tokens)
    )
    assert classify_legal_doc_type("01/2026/TT-BCA", "Thông tư 01") == "Thông tư"
    assert classify_legal_doc_type("168/2024/NĐ-CP", "Nghị định 168") == "Nghị định"
    assert classify_legal_doc_type("116/2025/QH15", "Luật An ninh mạng") == "Luật / Bộ luật"

    # Kiểm tra chuẩn hóa câu hỏi khẩu ngữ & số hiệu tắt
    norm_q, notes = normalize_legal_query_heuristic(
        "Đi xe đạp uống rượu bia bị phạt thế nào theo nghị định 168?",
        corpus_doc_numbers=["168/2024/NĐ-CP"],
    )
    assert "168/2024/NĐ-CP" in norm_q and "nồng độ cồn" in norm_q

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
    parser = argparse.ArgumentParser(
        description="Hệ thống Tìm kiếm Lai (BM25 + Dense + RRF) Văn bản Pháp luật & Quy chế Đào tạo"
    )
    parser.add_argument("--query", help="Câu hỏi tra cứu")
    parser.add_argument("--history", default="", help="Lịch sử hội thoại trước đó")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--fix-gpu",
        action="store_true",
        help="Tự động cài đặt lại PyTorch bản NVIDIA CUDA vào .venv hiện tại",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Kích hoạt kết nối Ollama cục bộ (qwen2.5:7b) cho Query Rewriting trên CLI",
    )
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if args.fix_gpu:
        fix_pytorch_cuda_installation()
        return

    if args.use_ollama:
        os.environ["USE_LOCAL_OLLAMA"] = "1"

    base_dir = get_app_base_dir()
    retriever = HybridRetriever(base_dir / "corpus.jsonl")
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
            base_dir / "queries.jsonl",
            base_dir / "evaluation",
            **search_options,
        )
        return
    query = rewrite_query(args.history, args.query or "đăng ký doanh nghiệp qua mạng")
    print(json.dumps({"original": args.query, "rewritten": query}, ensure_ascii=False))
    print(
        json.dumps(
            retriever.search(query, **search_options),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()