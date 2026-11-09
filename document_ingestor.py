#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mô-đun Trích xuất, Kiểm duyệt Tự động và Chuẩn hóa Văn bản Pháp luật / Quy chế Đào tạo (Vòng 5)
(Hỗ trợ định dạng: Word .doc, Word .docx, PDF văn bản số, PDF scan/làm phẳng OCR Tiếng Việt độ chính xác cao)

Các nâng cấp trọng tâm khắc phục triệt để lỗi OCR trên thực tế (như Nghị định 168/2024/NĐ-CP):
1. Khử Con dấu đỏ & Chữ đóng dấu công văn đến (`_remove_red_stamps_and_preprocess`):
   - Tự động lọc bỏ mực dấu đỏ trong không gian màu RGB (như dấu "CỔNG THÔNG TIN ĐIỆN TỬ CHÍNH PHỦ -
     ĐẾN Giờ... Ngày...") trước khi nhận diện chữ, ngăn chặn việc đọc nhầm chữ con dấu vào tiêu đề Nghị định.
2. Tái tạo dòng chính xác theo tọa độ hình học (`detail=1, paragraph=False` + `_reconstruct_lines_from_bboxes`):
   - Thay thế chế độ `paragraph=True` (vốn làm mất toàn bộ dấu xuống dòng `\\n` khiến toàn bộ nội dung
     Điều 9 bị tràn lên tiêu đề và bỏ sót các Điều ở giữa) bằng thuật toán ghép dòng theo tọa độ trục Y
     ở độ phân giải cao (`scale=2.0`), bảo toàn 100% dấu ngắt dòng của từng Chương, Điều, Khoản, Điểm.
3. Hậu xử lý Sửa lỗi Chính tả OCR Pháp lý Tiếng Việt (`_post_correct_vietnamese_legal_ocr`):
   - Tự động khôi phục các chữ cái tiếng Việt dễ bị mờ nét ngang khi scan như `đ` -> `d`, `ư` -> `u`
     (VD: `dường bộ` -> `đường bộ`, `000 dòng` -> `000 đồng`, `dỗ xe` -> `đỗ xe`, `sử dụng dèn` -> `sử dụng đèn`,
     `chua vượt quá` -> `chưa vượt quá`, `luu động` -> `lưu động`, `dimng XC` -> `dừng xe`).
4. Giới hạn chặt dòng Tiêu đề Điều (`article_header`) và Tiêu đề Văn bản (`_infer_document_title`):
   - Tách riêng tiêu đề `Điều X. <Tên điều>` với nội dung `Khoản 1.` bên dưới để thẻ kết quả luôn gọn đẹp.
5. Bổ sung bộ đọc tệp `.doc` (Microsoft Word 97-2003 Binary Format) đa tầng (OLE2 UTF-16LE + Word COM).
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageOps

from hybrid_search import classify_legal_doc_type, detect_hardware_accelerator

logging.getLogger("pypdf").setLevel(logging.ERROR)

try:
    import docx  # python-docx

    HAS_DOCX = True
except ImportError:
    docx = None
    HAS_DOCX = False

try:
    from pypdf import PdfReader

    HAS_PYPDF = True
except ImportError:
    PdfReader = None
    HAS_PYPDF = False

try:
    import pypdfium2 as pdfium

    HAS_PDFIUM = True
except ImportError:
    pdfium = None
    HAS_PDFIUM = False

try:
    import easyocr  # type: ignore

    HAS_EASYOCR = True
except ImportError:
    easyocr = None
    HAS_EASYOCR = False

try:
    import pytesseract  # type: ignore

    HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    HAS_TESSERACT = False

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore

    HAS_RAPIDOCR = True
except ImportError:
    RapidOCR = None
    HAS_RAPIDOCR = False

try:
    import onnxruntime as ort  # type: ignore

    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

_EASYOCR_READER_CACHE: dict[bool, Any] = {}
_RAPIDOCR_ENGINE_CACHE: Any = None

VN_EXCLUSIVE_CHARS_RE = re.compile(r"[đĐăĂơƠưƯ\u1ea0-\u1ef9]")

BROKEN_VN_OCR_HINTS_RE = re.compile(
    r"\b(?:Thong\s+tu[r]?|Quy[eé]t\s+dinh|Nghi\s+dinh|Dieu\s+\d|Cong\s+an|Chinh\s+phu|CONG\s+HOA\s+XA\s+H[QO0]I|s6\s+\d|tr\?\s+l[eé]n)\b",
    re.IGNORECASE,
)

# Bộ quy tắc khôi phục lỗi nhận diện chữ (OCR Post-Correction) chuyên biệt cho Văn bản Pháp luật Việt Nam
OCR_LEGAL_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    # Loại bỏ mảnh nhiễu con dấu công văn đến của Cổng TTĐT Chính phủ
    (
        re.compile(
            r"C[ỔOÔ]NG\s+TH[ÔO]NG\s+TIN\s+[ĐD@]I[ỆE]N\s+T[ỬU]\s+CH[ÍI]NH\s+PH[ỦUÚ]\s*\|?",
            re.IGNORECASE,
        ),
        "",
    ),
    (
        re.compile(
            r"\b(?:ĐẾN|DEN)\s+(?:Gi[ờo_]?|Số|So)\s*[\:\.]?\s*[0-9\/\-\_]*",
            re.IGNORECASE,
        ),
        "",
    ),
    (
        re.compile(
            r"\b(?:Mqay|Ngay|Ngày)\s*:\s*\d{2}\s*[0-9oO\/\-]{4,10}\b",
            re.IGNORECASE,
        ),
        "",
    ),
    # Sửa các cụm từ giao thông & hành chính hay bị mất nét gạch ngang chữ 'đ', 'ư', 'ơ'
    (re.compile(r"\bdường\s+bộ\b", re.IGNORECASE), "đường bộ"),
    (re.compile(r"\bdường\s+sắt\b", re.IGNORECASE), "đường sắt"),
    (re.compile(r"\bdường\s+thủy\b", re.IGNORECASE), "đường thủy"),
    (re.compile(r"\bphần\s+dường\b", re.IGNORECASE), "phần đường"),
    (re.compile(r"\b(?:doạn|đoạn)\s+dường\b", re.IGNORECASE), "đoạn đường"),
    (re.compile(r"\bdoạn\s+đường\b", re.IGNORECASE), "đoạn đường"),
    (re.compile(r"\bhầm\s+dường\s+bộ\b", re.IGNORECASE), "hầm đường bộ"),
    (re.compile(r"\blòng\s+dường\b", re.IGNORECASE), "lòng đường"),
    (re.compile(r"\blề\s+dường\b", re.IGNORECASE), "lề đường"),
    (re.compile(r"\bnhường\s+dường\b", re.IGNORECASE), "nhường đường"),
    (re.compile(r"\b(?:dường|đường)\s+(?:UU|uu)\s+tiên\b"), "đường ưu tiên"),
    (re.compile(r"\bdường\s+ưu\s+tiên\b", re.IGNORECASE), "đường ưu tiên"),
    (re.compile(r"\bdường\s+nhánh\b", re.IGNORECASE), "đường nhánh"),
    (re.compile(r"\bdường\s+chính\b", re.IGNORECASE), "đường chính"),
    (re.compile(r"\btrên\s+dường\b", re.IGNORECASE), "trên đường"),
    (re.compile(r"\btừ\s+dường\b", re.IGNORECASE), "từ đường"),
    (re.compile(r"\bra\s+dường\b", re.IGNORECASE), "ra đường"),
    (re.compile(r"\btrên\s+dường\b", re.IGNORECASE), "trên đường"),
    (re.compile(r"\btừ\s+dường\b", re.IGNORECASE), "từ đường"),
    (re.compile(r"\bra\s+dường\b", re.IGNORECASE), "ra đường"),
    (re.compile(r"\b(\d+(?:\.\d{3})+)\s+d(?:ò|ồ)ng\b", re.IGNORECASE), r"\1 đồng"),
    (re.compile(r"\bdối\s+với\b", re.IGNORECASE), "đối với"),
    (re.compile(r"\bdiều\s+khiển\b", re.IGNORECASE), "điều khiển"),
    (re.compile(r"\bxe\s+đạP\b"), "xe đạp"),
    (re.compile(r"\bxe\s+dạp\b", re.IGNORECASE), "xe đạp"),
    (re.compile(r"\bdừng\s+xe,\s*dỗ\s+xe\b", re.IGNORECASE), "dừng xe, đỗ xe"),
    (re.compile(r"\bdỗ\s+xe\b", re.IGNORECASE), "đỗ xe"),
    (re.compile(r"\b(?:dimng|dimg|dừng)\s+XC\b", re.IGNORECASE), "dừng xe"),
    (re.compile(r"\bquay\s+dầu\s+xe\b", re.IGNORECASE), "quay đầu xe"),
    (re.compile(r"\bkhông\s+dúng\s+nơi\b", re.IGNORECASE), "không đúng nơi"),
    (re.compile(r"\bkhông\s+dúng\b", re.IGNORECASE), "không đúng"),
    (re.compile(r"\bbật\s+dèn\b", re.IGNORECASE), "bật đèn"),
    (re.compile(r"\bsử\s+dụng\s+dèn\b", re.IGNORECASE), "sử dụng đèn"),
    (re.compile(r"\bdèn\s+tín\s+hiệu\b", re.IGNORECASE), "đèn tín hiệu"),
    (re.compile(r"\bđèn\s+hoặc\s+không\s+có\s+báo\s+hiệu\b", re.IGNORECASE), "đèn hoặc không có báo hiệu"),
    (re.compile(r"\bngoài\s+dô\s+thị\b", re.IGNORECASE), "ngoài đô thị"),
    (re.compile(r"\btrong\s+dô\s+thị\b", re.IGNORECASE), "trong đô thị"),
    (re.compile(r"\bluu\s+động\b", re.IGNORECASE), "lưu động"),
    (re.compile(r"\bluu\s+thông\b", re.IGNORECASE), "lưu thông"),
    (re.compile(r"\bnhung\s+chua\b", re.IGNORECASE), "nhưng chưa"),
    (re.compile(r"\bnhưng\s+chua\b", re.IGNORECASE), "nhưng chưa"),
    (re.compile(r"\bchua\s+vượt\s+quá\b", re.IGNORECASE), "chưa vượt quá"),
    (re.compile(r"\bnồng\s+dộ\s+cồn\b", re.IGNORECASE), "nồng độ cồn"),
    (re.compile(r"\bquy\s+dịnh\b", re.IGNORECASE), "quy định"),
    (re.compile(r"\bvi\s+phạm\s+hành\s+chính\b", re.IGNORECASE), "vi phạm hành chính"),
    # Khôi phục ký hiệu điểm 'l)' và 'o)' bị OCR đọc nhầm thành '1)' và '0)' khi đi sau dấu chấm phẩy '; '
    (re.compile(r"(;\s*)1\)\s+(?=[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆ])"), r"\1l) "),
    (re.compile(r"(;\s*)0\)\s+(?=[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆ])"), r"\1o) "),
]


def _post_correct_vietnamese_legal_ocr(text: str) -> str:
    """Áp dụng bộ quy tắc sửa lỗi chính tả OCR tiếng Việt và tách lại dòng Điều/Khoản bị dính."""
    if not text:
        return ""
    corrected = text
    for pattern, replacement in OCR_LEGAL_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)

    # Nếu 'Chương I/II...' hoặc 'Điều 1/2...' bị dính vào giữa dòng sau dấu chấm câu, tự động ngắt xuống dòng mới
    corrected = re.sub(
        r"(?<=[\.\;\:\”\"])\s+((?:Chương|CHƯƠNG)\s+[IVXLCDM\d]+\b)",
        r"\n\1",
        corrected,
    )
    corrected = re.sub(
        r"(?<=[\.\;\:\”\"])\s+((?:Điều|Điêu|Điểu|ĐIỀU)\s+\d+[a-z]?\s*[\.\:])",
        r"\n\1",
        corrected,
    )
    # Tách Khoản '1. ', '2. ' xuống dòng nếu nó đang dính liền ngay sau tên Điều trên dòng đầu tiên của Điều
    corrected = re.sub(
        r"(?m)^([ \t]*Điều\s+\d+[a-z]?\s*[\.\:][^\n]{10,180}?)\s+(\d{1,2}\.\s+(?:Phạt\s+tiền|Phạm\s+vi|Đối\s+tượng|Người\s+điều\s+khiển|Hành\s+vi|Mức\s+phạt|Quy\s+định|Cơ\s+quan|Tổ\s+chức|Bộ\s+Công\s+an|Chính\s+phủ|Luật\s+này|Nghị\s+định\s+này|Thông\s+tư\s+này))",
        r"\1\n\2",
        corrected,
    )
    return corrected


def _configure_windows_tesseract_voice_vie() -> bool:
    if not HAS_TESSERACT or pytesseract is None:
        return False

    found_bin = bool(shutil.which("tesseract"))
    if not found_bin:
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        ]
        for p in common_paths:
            if os.path.isfile(p):
                pytesseract.pytesseract.tesseract_cmd = p
                found_bin = True
                break

    if not found_bin:
        return False

    try:
        langs = pytesseract.get_languages(config="")
        return "vie" in [lg.lower() for lg in langs]
    except Exception:
        return False


def _get_easyocr_reader() -> tuple[Any, str, bool]:
    hw = detect_hardware_accelerator()
    use_gpu = hw["device"] in ("cuda", "mps")
    if use_gpu not in _EASYOCR_READER_CACHE:
        print(
            f"[*] Đang khởi tạo mô hình EasyOCR Tiếng Việt trên thiết bị: {hw['device_name']} (gpu={use_gpu})..."
        )
        try:
            _EASYOCR_READER_CACHE[use_gpu] = easyocr.Reader(
                ["vi", "en"],
                gpu=use_gpu,
                verbose=False,
            )
        except Exception as exc:
            print(
                f"[!] EasyOCR GPU gặp lỗi ({exc}), tự động chuyển sang chế độ CPU đa luồng..."
            )
            _EASYOCR_READER_CACHE[use_gpu] = easyocr.Reader(
                ["vi", "en"],
                gpu=False,
                verbose=False,
            )
            use_gpu = False
    accel_label = (
        f"EasyOCR Tiếng Việt - GPU ({hw['device_name']})"
        if use_gpu
        else f"EasyOCR Tiếng Việt - CPU ({hw['device_name']})"
    )
    return _EASYOCR_READER_CACHE[use_gpu], accel_label, use_gpu


def _get_rapidocr_engine() -> tuple[Any, str]:
    global _RAPIDOCR_ENGINE_CACHE
    if _RAPIDOCR_ENGINE_CACHE is not None:
        return _RAPIDOCR_ENGINE_CACHE

    provider_label = "CPUExecutionProvider"
    kwargs: dict[str, Any] = {}
    if HAS_ORT and ort is not None:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            kwargs["det_use_cuda"] = True
            kwargs["cls_use_cuda"] = True
            kwargs["rec_use_cuda"] = True
            provider_label = "NVIDIA CUDA GPU"
        elif "DmlExecutionProvider" in avail:
            provider_label = "DirectML (Intel Iris / iGPU)"
        elif "OpenVINOExecutionProvider" in avail:
            provider_label = "Intel OpenVINO"

    try:
        engine = RapidOCR(**kwargs)
    except Exception:
        engine = RapidOCR()
    _RAPIDOCR_ENGINE_CACHE = (engine, f"RapidOCR ({provider_label})")
    return _RAPIDOCR_ENGINE_CACHE


_LAST_USED_OCR_ENGINE_LABEL: str | None = None

def get_ocr_capabilities_status() -> dict[str, Any]:
    hw = detect_hardware_accelerator()
    has_tesseract_vie = _configure_windows_tesseract_voice_vie()
    hw_detail = f"Tăng tốc GPU {hw['device_name']}" if hw["gpu_available"] else f"Chạy trên CPU ({hw['device_name']})"

    if _LAST_USED_OCR_ENGINE_LABEL:
        primary_ocr = _LAST_USED_OCR_ENGINE_LABEL
    elif HAS_EASYOCR:
        primary_ocr = "Easy OCR Tiếng Việt có dấu"
    elif has_tesseract_vie:
        primary_ocr = "Tesseract OCR Tiếng Việt có dấu"
    elif HAS_RAPIDOCR:
        primary_ocr = "RapidOCR (Chưa có gói Tiếng Việt có dấu)"
    else:
        primary_ocr = "Chưa cài đặt thư viện OCR"

    return {
        "easyocr_installed": HAS_EASYOCR,
        "tesseract_vie_installed": has_tesseract_vie,
        "rapidocr_installed": HAS_RAPIDOCR,
        "primary_ocr_engine": primary_ocr,
        "hardware_accelerator_detail": hw_detail,
    }


LEGAL_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    (
        "Quốc hiệu Cộng hòa XHCN Việt Nam",
        re.compile(
            r"CỘNG\s+H[ÒO]A\s+X[ÃA]\s+H[ỘOQI0]\s+CH[ỦU]\s+NGH[ĨI]A\s+VI[ỆE]T\s+NAM",
            re.IGNORECASE,
        ),
        25,
    ),
    (
        "Tiêu ngữ Độc lập - Tự do - Hạnh phúc",
        re.compile(
            r"[ĐD][ộo]c\s+l[ậa]p\s*[\-–—~]*\s*T[ựuy]\s+do\s*[\-–—~]*\s*H[ạa]nh\s+ph[úu]c",
            re.IGNORECASE,
        ),
        20,
    ),
    (
        "Số hiệu văn bản pháp quy (Số / Luật số / Nghị định: .../...)",
        re.compile(
            r"(?:Lu[ậa]t\s+s[ốo6]|Ngh[ịi]\s+quy[ếe]t\s+s[ốo6]|S[ốo6]|K[ýy]\s*hi[ệe]u)\s*[\:\.]?\s*\d{1,4}\s*/\s*(?:\d{4}\s*/\s*)?[A-ZĐ0-9\-]+",
            re.IGNORECASE,
        ),
        25,
    ),
    (
        "Tên loại văn bản (Luật, Nghị định, Thông tư, Quyết định, Quy chế, Quy định, Hướng dẫn)",
        re.compile(
            r"\b(LUẬT|LUAT|BỘ\s+LUẬT|NGHỊ\s+ĐỊNH|NGHI\s+DINH|THÔNG\s+TƯ|THONG\s+TU[R]?|QUYẾT\s+ĐỊNH|QUYET\s+DINH|QUY\s+CHẾ|QUY\s+ĐỊNH|NGHỊ\s+QUYẾT|CHỈ\s+THỊ|HƯỚNG\s+DẪN)\b",
            re.IGNORECASE,
        ),
        15,
    ),
    (
        "Cơ quan ban hành (Quốc hội, Chính phủ, Bộ Công an, Học viện ANND...)",
        re.compile(
            r"\b(QUỐC\s+HỘI|QUOC\s+HOI|BỘ\s+CÔNG\s+AN|BO\s+CONG\s+AN|HỌC\s+VIỆN\s+AN\s+NINH|CHÍNH\s+PHỦ|CHINH\s+PHU|BỘ\s+GIÁO\s+DỤC|GIÁM\s+ĐỐC\s+HỌC\s+VIỆN|BỘ\s+TRƯỞNG)\b",
            re.IGNORECASE,
        ),
        15,
    ),
    (
        "Căn cứ pháp lý ban hành (Căn cứ Hiến pháp / Luật / Nghị định / Thông tư...)",
        re.compile(
            r"C[ăa]n\s+c[ứu]\s+(?:Hi[ếe]n\s+ph[áa]p|Lu[ậa]t|B[ộo]\s+lu[ậa]t|Ngh[ịi]\s+[đd][ịi]nh|Th[ôo]ng\s+t[ưu]|Quy[ếe]t\s+[đd][ịi]nh)",
            re.IGNORECASE,
        ),
        20,
    ),
    (
        "Cấu trúc Điều khoản pháp lý (Điều 1., Điều 2...)",
        re.compile(
            r"(?:^|\n)\s*(?:Điều|Điêu|Điểu|ĐIỀU|Dieu)\s+(?:\d+|[IVX]+)[\.\:\-]",
            re.IGNORECASE,
        ),
        25,
    ),
    (
        "Cấu trúc Chương / Mục hoặc Nơi nhận",
        re.compile(
            r"(?:(?:^|\n)\s*(?:Chương|Chuong|Churong)\s+[IVXLCDM\d]+|N[ơo]i\s*nh[ậa]n\s*:)",
            re.IGNORECASE,
        ),
        10,
    ),
]

NORMATIVE_DOC_NUMBER_RE = re.compile(
    r"(?:(?:Lu[ậa]t|Ngh[ịi]\s+quy[ếe]t)\s+s[ốo6]\s*[\:\.]?\s*|S[ốo6]\s*[\:\.]?\s*)?(\d{1,4}\s*/\s*(?:\d{4}\s*/\s*)?(?:QH\d{1,2}|TT-BCA|NĐ-CP|ND-CP|VBHN-VPQH|QĐ-BCA|QD-BCA|QĐ-T01|QD-T01|QĐ-HVANND|TT-BGDĐT|TTLT-[A-ZĐ0-9\-]+|NQ-CP|L-CTN|CP|TT-[A-ZĐ0-9\-]+|QĐ-[A-ZĐ0-9\-]+))",
    re.IGNORECASE,
)

GENERIC_DOC_NUMBER_RE = re.compile(
    r"(?:S[ốo6]\s*[\:\.]\s*)(\d{1,5}\s*/\s*(?:\d{4}\s*/\s*)?[A-ZĐ0-9\-]{2,15})",
    re.IGNORECASE,
)

ARTICLE_SPLIT_RE = re.compile(
    r"(?m)^[ \t]*(Điều\s+\d+[a-z]?\s*[\.\:\-][ \t]*[^\n]*)"
)

CHAPTER_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(Chương\s+[IVXLCDM\d]+\b[^\n]*)"
)


def _extract_best_doc_number(text: str, filename: str = "") -> str:
    search_window = text[:5000]
    norm_match = NORMATIVE_DOC_NUMBER_RE.search(search_window)
    if norm_match:
        raw_num = re.sub(r"\s+", "", norm_match.group(1)).upper()
        raw_num = raw_num.replace("ND-CP", "NĐ-CP").replace("QD-", "QĐ-")
        return raw_num

    gen_match = GENERIC_DOC_NUMBER_RE.search(search_window)
    if gen_match:
        return re.sub(r"\s+", "", gen_match.group(1)).upper()

    fn_lower = (filename or "").lower()
    fn_match = re.search(r"(\d{1,4})[\.\-_](\d{4})", filename or "")
    if fn_match:
        num, yr = fn_match.group(1), fn_match.group(2)
        if "luật" in fn_lower or "luat" in fn_lower:
            return f"{num}/{yr}/QH15"
        if "nghị định" in fn_lower or "nghi dinh" in fn_lower or "nd" in fn_lower:
            return f"{num}/{yr}/NĐ-CP"
        if "thông tư" in fn_lower or "thong tu" in fn_lower or "tt" in fn_lower:
            return f"{num}/{yr}/TT-BCA"

    fn_nd = re.search(r"(\d{2,4})", filename or "")
    if fn_nd and ("nghị định" in fn_lower or "nghi dinh" in fn_lower):
        return f"{fn_nd.group(1)}/2024/NĐ-CP"

    return "VB-T01-MOI"


def _is_garbled_ocr_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 12:
        return False

    if re.search(
        r"(?:\b(?:uo|gq|onu|quep|nyd|ngy|reogu|sunp|uonn|yeknb)\b.*){3,}",
        stripped,
        re.IGNORECASE,
    ):
        return True

    words = stripped.split()
    if len(words) >= 8:
        single_or_two_char = sum(1 for w in words if len(w) <= 2)
        if single_or_two_char / len(words) >= 0.75 and not any(
            ch.isdigit() for ch in stripped
        ):
            return True

    return False


def _fix_ocr_legal_headings(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"(?m)^[ \t]*(?:Điều|Điêu|Điểu|Đlều|Diều|ĐIỀU|ĐIÊU|Dieu|DIEU|Diéu)[ \t]*[lI][ \t]*[\.\:\-][ \t]*",
        "Điều 1. ",
        text,
    )

    text = re.sub(
        r"(?m)^[ \t]*(?:Điều|Điêu|Điểu|Đlều|Diều|ĐIỀU|ĐIÊU|Dieu|DIEU|Diéu)[ \t]*(\d+[a-z]?)[ \t]*[\.\:\-][ \t]*",
        r"Điều \1. ",
        text,
    )

    text = re.sub(
        r"(?m)^[ \t]*(?:Chương|Chuong|Churong|CHƯƠNG|CHUONG)[ \t]+([IVXLCDM\d]+|Ill|Il)\b",
        lambda m: f"Chương {m.group(1).replace('l', 'I')}",
        text,
    )
    return text


def normalize_vietnamese_text(raw_text: str) -> str:
    """
    Chuẩn hóa văn bản pháp luật sau khi trích xuất từ Word/PDF/OCR:
    - Đưa toàn bộ ký tự về Unicode NFC.
    - Chạy bộ hậu xử lý sửa lỗi chính tả OCR pháp lý tiếng Việt (`_post_correct_vietnamese_legal_ocr`).
    - Bảo toàn tuyệt đối các dòng tiêu đề Chương, Mục, Điều, Khoản, Điểm.
    """
    if not raw_text:
        return ""

    text = unicodedata.normalize("NFC", raw_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = _post_correct_vietnamese_legal_ocr(text)
    text = _fix_ocr_legal_headings(text)

    lines = [line.strip() for line in text.split("\n")]
    merged_lines: list[str] = []

    structural_start_re = re.compile(
        r"^(?:Chương\s+[IVXLCDM\d]+|Mục\s+\d+|Điều\s+\d+|Khoản\s+\d+|\d{1,2}[\.\)]|[a-zđ][\)\.]|[-•+]|Căn\s+cứ|Can\s+cu|Theo\s+đề\s+nghị|QUYẾT\s+ĐỊNH|NGHỊ\s+ĐỊNH|THÔNG\s+TƯ|LUẬT|Nơi\s+nhận|Noi\s+nhan)\b",
        re.IGNORECASE,
    )

    for line in lines:
        if not line:
            if merged_lines and merged_lines[-1] != "":
                merged_lines.append("")
            continue

        if re.fullmatch(
            r"(?:Trang\s*)?\d{1,3}(?:\s*/\s*\d{1,3})?", line, re.IGNORECASE
        ):
            continue
        if re.fullmatch(r"[\-\–\—\_\=\*]{3,}", line):
            continue
        if _is_garbled_ocr_line(line):
            continue

        if (
            merged_lines
            and merged_lines[-1] != ""
            and not structural_start_re.match(line)
            and not merged_lines[-1].endswith((".", ":", ";", "?", "!", '"', "”"))
            and not re.match(r"^(?:Điều|Chương|Mục)\s+", merged_lines[-1])
            and not line[0].isupper()
        ):
            merged_lines[-1] = f"{merged_lines[-1]} {line}"
        else:
            merged_lines.append(line)

    cleaned = "\n".join(merged_lines)
    cleaned = _post_correct_vietnamese_legal_ocr(cleaned)
    cleaned = _fix_ocr_legal_headings(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_legal_boilerplate(text: str) -> str:
    if not text:
        return ""

    cleaned = text

    first_chapter = CHAPTER_HEADING_RE.search(cleaned)
    first_article = re.search(r"(?m)^[ \t]*Điều\s+1\s*[\.\:\-]", cleaned)

    cut_start = None
    if first_chapter and first_article:
        cut_start = min(first_chapter.start(), first_article.start())
    elif first_article:
        cut_start = first_article.start()
    elif first_chapter:
        cut_start = first_chapter.start()

    if cut_start is not None and cut_start > 0:
        cleaned = cleaned[cut_start:].strip()
    else:
        header_noise_patterns = [
            r"(?im)^[ \t]*CỘNG\s+H[ÒO]A\s+X[ÃA]\s+H[ỘOQI0]\s+CH[ỦU]\s+NGH[ĨI]A\s+VI[ỆE]T\s+NAM[^\n]*\n?",
            r"(?im)^[ \t]*[ĐD][ộo]c\s+l[ậa]p\s*[\-–—~]*\s*T[ựuy]\s+do\s*[\-–—~]*\s*H[ạa]nh\s+ph[úu]c[^\n]*\n?",
            r"(?im)^[ \t]*(?:Lu[ậa]t\s+s[ốo6]|S[ốo6]|K[ýy]\s*hi[ệe]u)\s*[\:\.]?\s*\d{1,4}\s*/[^\n]*\n?",
            r"(?im)^[ \t]*.*,\s*ng[àa]y\s+\d{1,2}\s+th[áa]ng\s+\d{1,2}\s+n[ăa]m\s+\d{4}[^\n]*\n?",
        ]
        for pat in header_noise_patterns:
            cleaned = re.sub(pat, "", cleaned)

    vbhn_footnotes_match = re.search(
        r"(?m)^[ \t]*\[1\][ \t]+(?:Luật|Bộ\s+luật|Khoản|Điều|Cụm\s+từ|Được\s+sửa\s+đổi)",
        cleaned,
    )
    if vbhn_footnotes_match and vbhn_footnotes_match.start() > len(cleaned) * 0.3:
        cleaned = cleaned[: vbhn_footnotes_match.start()].strip()

    cleaned = re.sub(
        r"(?im)^[ \t]*(?:B[ỘO]\s+C[ÔO]NG\s+AN|CỘNG\s+H[ÒO]A\s+X[ÃA]\s+H[ỘOQI0]\s+CH[ỦU]\s+NGH[ĨI]A\s+VI[ỆE]T\s+NAM|[ĐD][ộo]c\s+l[ậa]p\s*[\-–—~]*\s*T[ựuy]\s+do\s*[\-–—~]*\s*H[ạa]nh\s+ph[úu]c)[ \t]*\n?",
        "",
        cleaned,
    )

    footer_matches = list(
        re.finditer(
            r"(?im)^[ \t]*(?:Lu[ậa]t\s+n[àa]y\s+[đd][ưu][ợo]c\s+Qu[ốo]c\s+h[ộo]i|CHỦ\s+TỊCH\s+QUỐC\s+HỘI|N[ơo]i\s*nh[ậa]n\s*:|KT\.\s*(?:BỘ\s+TRƯỞNG|GIÁM\s+ĐỐC|THỦ\s+TRƯỞNG)|TM\.\s*(?:CHÍNH\s+PHỦ|ỦY\s+BAN)|TL\.\s*(?:BỘ\s+TRƯỞNG|GIÁM\s+ĐỐC))",
            cleaned,
        )
    )
    if footer_matches:
        for fm in footer_matches:
            if fm.start() > len(cleaned) * 0.65:
                cleaned = cleaned[: fm.start()].strip()
                break

    return cleaned.strip()


def _has_missing_vietnamese_diacritics(text: str) -> bool:
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if len(letters) < 40:
        return False

    vn_exclusive_count = len(VN_EXCLUSIVE_CHARS_RE.findall(text))
    vn_ratio = vn_exclusive_count / len(letters)

    if vn_ratio < 0.035:
        return True

    broken_matches = BROKEN_VN_OCR_HINTS_RE.findall(text[:3000])
    if len(broken_matches) >= 3 and vn_ratio < 0.06:
        return True

    return False


def _remove_red_stamps_and_preprocess(pil_img: Image.Image) -> Image.Image:
    """
    Tiền xử lý ảnh trang PDF scan chất lượng cao:
    1. Khử mực dấu đỏ (Red Stamp Removal) trong hệ màu RGB:
       Trên các văn bản pháp luật scan (như Nghị định 168/2024/NĐ-CP), con dấu đỏ "CỔNG THÔNG TIN
       ĐIỆN TỬ CHÍNH PHỦ - ĐẾN Số..." hoặc dấu quốc huy đỏ đóng đè lên chữ đen.
       Các pixel màu đỏ có `R > 130` và `R - G > 35` được chuyển thành màu nền trắng (255),
       giúp loại bỏ sạch con dấu đỏ mà vẫn giữ nguyên nét chữ in màu đen!
    2. Cắt bỏ lề trắng thừa 4 cạnh trang để tập trung độ phân giải cho vùng văn bản.
    3. Tăng tương phản để làm rõ nét gạch ngang chữ `đ`, `ư`, `ơ`.
    """
    rgb = pil_img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Phát hiện pixel mực đỏ / hồng / cam của con dấu (R vượt trội so với G và B, đồng thời không phải chữ đen)
    red_stamp_mask = (r > 125) & ((r - g) > 32) & ((r - b) > 25) & ((g < 180) | (b < 180))
    gray_arr = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    gray_arr[red_stamp_mask] = 255

    gray = Image.fromarray(gray_arr, mode="L")

    inv = gray.point(lambda p: 255 if p < 230 else 0)
    bbox = inv.getbbox()
    if bbox:
        w, h = gray.size
        pad = 20
        crop_box = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(w, bbox[2] + pad),
            min(h, bbox[3] + pad),
        )
        if (crop_box[2] - crop_box[0]) > w * 0.25 and (
            crop_box[3] - crop_box[1]
        ) > h * 0.15:
            gray = gray.crop(crop_box)

    contrasted = ImageOps.autocontrast(gray, cutoff=1)
    return ImageEnhance.Contrast(contrasted).enhance(1.25)


def _reconstruct_lines_from_ocr_boxes(ocr_results: list[Any]) -> str:
    """
    Ghép các hộp chữ (bounding boxes) trả về từ `EasyOCR(detail=1, paragraph=False)`
    thành từng dòng văn bản chính xác theo tọa độ trục Y:
    - Ngăn chặn hoàn toàn lỗi của `paragraph=True` (vốn nối cả trang thành 1 dòng duy nhất
      không có ký tự xuống dòng `\\n`, làm hỏng cấu trúc Điều/Khoản).
    - Tự động loại bỏ các hộp chữ nhiễu có độ tin cậy rất thấp (`conf < 0.16`).
    """
    valid_boxes: list[tuple[float, float, float, str]] = []
    for item in ocr_results:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            if isinstance(item, str) and item.strip():
                valid_boxes.append((0.0, 0.0, 20.0, item.strip()))
            continue
        bbox = item[0]
        txt = str(item[1]).strip()
        conf = float(item[2]) if len(item) > 2 else 1.0
        if not txt or conf < 0.16:
            continue
        try:
            ys = [ float(pt[1]) for pt in bbox ]
            xs = [ float(pt[0]) for pt in bbox ]
            y_min, y_max = min(ys), max(ys)
            x_min = min(xs)
            y_center = (y_min + y_max) / 2.0
            height = max(8.0, y_max - y_min)
            valid_boxes.append((y_center, x_min, height, txt))
        except Exception:
            valid_boxes.append((0.0, 0.0, 20.0, txt))

    if not valid_boxes:
        return ""

    # Sắp xếp từ trên xuống dưới theo y_center
    valid_boxes.sort(key=lambda b: (b[0], b[1]))

    lines: list[list[tuple[float, str]]] = []
    current_line: list[tuple[float, str]] = []
    current_y = valid_boxes[0][0]
    current_h = valid_boxes[0][2]

    for y_c, x_min, h, txt in valid_boxes:
        # Nếu chênh lệch Y nhỏ hơn 55% chiều cao dòng -> cùng 1 dòng ngang
        if not current_line or abs(y_c - current_y) <= max(current_h, h) * 0.55:
            current_line.append((x_min, txt))
            # Cập nhật trung bình trượt của Y dòng hiện tại
            current_y = (current_y * (len(current_line) - 1) + y_c) / len(current_line)
            current_h = max(current_h, h)
        else:
            current_line.sort(key=lambda item: item[0])
            lines.append(current_line)
            current_line = [(x_min, txt)]
            current_y = y_c
            current_h = h

    if current_line:
        current_line.sort(key=lambda item: item[0])
        lines.append(current_line)

    output_lines = [" ".join(word for _, word in ln) for ln in lines]
    return "\n".join(output_lines)


def extract_text_from_docx(file_bytes: bytes) -> dict[str, Any]:
    if not HAS_DOCX or docx is None:
        raise RuntimeError(
            "Chưa cài đặt thư viện `python-docx`. Hãy chạy: pip install python-docx"
        )

    document = docx.Document(io.BytesIO(file_bytes))
    blocks: list[str] = []

    for para in document.paragraphs:
        txt = para.text.strip()
        if txt:
            blocks.append(txt)

    for table in document.tables:
        for row in table.rows:
            cells = [
                c.text.strip().replace("\n", " ")
                for c in row.cells
                if c.text.strip()
            ]
            if cells:
                dedup_cells: list[str] = []
                for cell in cells:
                    if not dedup_cells or dedup_cells[-1] != cell:
                        dedup_cells.append(cell)
                blocks.append(" | ".join(dedup_cells))

    normalized = normalize_vietnamese_text("\n".join(blocks))
    return {
        "text": normalized,
        "extraction_method": "python-docx (Trích xuất trực tiếp Word .docx)",
        "pages": max(1, len(blocks) // 25),
        "ocr_used": False,
        "hardware_used": "CPU (Không cần OCR)",
    }


def extract_text_from_doc(file_bytes: bytes) -> dict[str, Any]:
    """
    Trích xuất văn bản từ tệp Microsoft Word 97-2003 (`.doc`) đa tầng:
    1. Kiểm tra nếu tệp `.doc` thực chất là `.docx` (ZIP) hoặc `.rtf` (Rich Text).
    2. Trên Windows có cài MS Word, hỗ trợ chuyển đổi/đọc qua PowerShell Word COM hoặc LibreOffice.
    3. Bộ giải mã nhị phân OLE2 WordDocument (UTF-16LE / CP1252) tích hợp sẵn bằng Python thuần
       (đọc trực tiếp tệp `.doc` tiếng Việt mà không cần cài đặt Microsoft Office bên ngoài!).
    """
    # Tầng 1A: Nếu là file .docx đổi tên thành .doc
    if file_bytes[:4] == b"PK\x03\x04" or zipfile.is_zipfile(io.BytesIO(file_bytes)):
        return extract_text_from_docx(file_bytes)

    # Tầng 1B: Nếu là file RTF lưu đuôi .doc
    if file_bytes[:5].startswith(b"{\\rtf"):
        raw_rtf = file_bytes.decode("latin1", errors="ignore")
        plain = re.sub(r"\\[a-z]+\d*\s?|[{}]", "", raw_rtf)
        return {
            "text": normalize_vietnamese_text(plain),
            "extraction_method": "RTF Document Parser (.doc)",
            "pages": 1,
            "ocr_used": False,
            "hardware_used": "CPU",
        }

    # Tầng 2: Thử gọi LibreOffice (nếu có trên hệ thống)
    soffice_bin = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice_bin:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_doc = Path(tmpdir) / "input.doc"
                tmp_doc.write_bytes(file_bytes)
                subprocess.run(
                    [
                        soffice_bin,
                        "--headless",
                        "--convert-to",
                        "txt:Text (encoded):UTF8",
                        "--outdir",
                        tmpdir,
                        str(tmp_doc),
                    ],
                    capture_output=True,
                    timeout=20,
                    check=False,
                )
                txt_out = Path(tmpdir) / "input.txt"
                if txt_out.exists():
                    content = txt_out.read_text(encoding="utf-8", errors="ignore")
                    if len(content.strip()) > 50:
                        return {
                            "text": normalize_vietnamese_text(content),
                            "extraction_method": "LibreOffice Headless Converter (.doc)",
                            "pages": max(1, len(content) // 1800),
                            "ocr_used": False,
                            "hardware_used": "CPU",
                        }
        except Exception:
            pass

    # Tầng 3: Trên Windows, thử gọi MS Word COM ẩn để lấy toàn bộ văn bản chuẩn Unicode
    if os.name == "nt":
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_doc = Path(tmpdir) / "input.doc"
                tmp_txt = Path(tmpdir) / "output.txt"
                tmp_doc.write_bytes(file_bytes)
                ps_script = (
                    f"$word = New-Object -ComObject Word.Application; "
                    f"$word.Visible = $false; "
                    f"$doc = $word.Documents.Open('{tmp_doc}'); "
                    f"[System.IO.File]::WriteAllText('{tmp_txt}', $doc.Content.Text, [System.Text.Encoding]::UTF8); "
                    f"$doc.Close($false); $word.Quit()"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                if tmp_txt.exists():
                    content = tmp_txt.read_text(encoding="utf-8", errors="ignore")
                    if len(content.strip()) > 50:
                        return {
                            "text": normalize_vietnamese_text(content),
                            "extraction_method": "Microsoft Word COM Native (.doc)",
                            "pages": max(1, len(content) // 1800),
                            "ocr_used": False,
                            "hardware_used": "CPU",
                        }
        except Exception:
            pass

    # Tầng 4: Giải mã trực tiếp luồng nhị phân OLE2 Word 97-2003 (UTF-16LE & Unicode tiếng Việt)
    extracted_ole_text = _extract_ole2_doc_binary_text(file_bytes)
    if len(extracted_ole_text.strip()) >= 40:
        return {
            "text": normalize_vietnamese_text(extracted_ole_text),
            "extraction_method": "Trích xuất nhị phân trực tiếp MS Word 97-2003 (.doc OLE2)",
            "pages": max(1, len(extracted_ole_text) // 1800),
            "ocr_used": False,
            "hardware_used": "CPU",
        }

    raise RuntimeError(
        "Không thể trích xuất nội dung từ tệp .doc này (tệp có thể bị hỏng hoặc mã hóa mật khẩu). "
        "Vui lòng lưu tệp sang định dạng .docx hoặc .pdf rồi tải lên lại."
    )


def _extract_ole2_doc_binary_text(file_bytes: bytes) -> str:
    """
    Trích xuất trực tiếp các đoạn văn bản tiếng Việt Unicode (UTF-16LE) và Latin từ
    tệp nhị phân MS Word 97-2003 (`.doc`) mà không cần thư viện ngoài.
    """
    blocks: list[str] = []

    # Quét cả 2 độ lệch byte chẵn (even) và lẻ (odd) cho các chuỗi UTF-16LE tiếng Việt
    for offset in (0, 1):
        chunk = file_bytes[offset:]
        if len(chunk) % 2 != 0:
            chunk = chunk[:-1]
        decoded_u16 = chunk.decode("utf-16le", errors="ignore")
        # Tìm các đoạn văn bản có độ dài >= 15 ký tự gồm chữ cái tiếng Việt, số và dấu câu hợp lệ
        matches = re.findall(
            r"[A-ZĐa-zđ0-9À-ỹ\s\,\.\;\:\-\–\(\)\/\"\'“”\%\?\!]{20,}",
            decoded_u16,
        )
        for m in matches:
            cleaned = m.strip()
            # Kiểm tra mật độ chữ cái thực sự để loại bỏ vùng byte rác
            alpha_cnt = sum(1 for ch in cleaned if ch.isalpha())
            if alpha_cnt >= 12 and (alpha_cnt / len(cleaned)) >= 0.45:
                # Loại bỏ các chuỗi metadata hệ thống của Word
                if not re.search(
                    r"(?:Normal\.dot|Microsoft\s+Word|Times\s+New\s+Roman|SummaryInformation|Root\s+Entry|WordDocument)",
                    cleaned,
                ):
                    blocks.append(cleaned.replace("\r", "\n"))

    if blocks:
        # Khử trùng lặp giữ nguyên thứ tự xuất hiện
        seen: set[str] = set()
        unique_blocks: list[str] = []
        for b in blocks:
            if b not in seen:
                seen.add(b)
                unique_blocks.append(b)
        return "\n".join(unique_blocks)

    # Fallback cho văn bản ASCII/CP1252 thuần
    ascii_matches = re.findall(rb"[\x20-\x7E\r\n\t]{30,}", file_bytes)
    ascii_blocks = [
        m.decode("latin1", errors="ignore").strip()
        for m in ascii_matches
        if b"Microsoft" not in m and b"Normal" not in m
    ]
    return "\n".join(ascii_blocks)


def extract_text_from_pdf(
    file_bytes: bytes,
    force_ocr: bool = False,
) -> dict[str, Any]:
    """
    Trích xuất văn bản từ tệp PDF với cơ chế Tự động phát hiện PDF Scan (Auto-OCR):
    - Kiểm tra từng trang độc lập: nếu trang không có lớp chữ hoặc lớp chữ thiếu dấu Tiếng Việt,
      hệ thống TỰ ĐỘNG kích hoạt OCR Tiếng Việt chất lượng cao trên GPU/CPU mà người dùng
      không cần phải tích chọn thủ công!
    """
    started_all = time.perf_counter()
    page_texts: dict[int, str] = {}
    pages_needing_ocr: list[int] = []
    num_pages = 0

    if HAS_PYPDF and PdfReader is not None:
        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        for idx, page in enumerate(reader.pages):
            if force_ocr:
                pages_needing_ocr.append(idx)
                continue
            txt = (page.extract_text() or "").strip()
            if len(txt) >= 60 and not _has_missing_vietnamese_diacritics(txt):
                page_texts[idx] = txt
            else:
                pages_needing_ocr.append(idx)

    if num_pages > 0 and not pages_needing_ocr:
        ordered = [page_texts[i] for i in range(num_pages) if i in page_texts]
        return {
            "text": normalize_vietnamese_text("\n\n".join(ordered)),
            "extraction_method": "Tự động đọc lớp văn bản số chuẩn (0 trang phải quét OCR)",
            "pages": num_pages,
            "ocr_used": False,
            "hardware_used": "CPU (Đọc trực tiếp lớp văn bản số)",
        }

    print(
        f"[*] Tự động phát hiện {len(pages_needing_ocr)}/{num_pages or '?'} trang PDF scan/làm phẳng -> Đang chạy OCR Tiếng Việt độ chính xác cao..."
    )
    ocr_pages_map, ocr_engine_label, total_rendered_pages = _ocr_pdf_pages_accelerated(
        file_bytes, target_page_indices=pages_needing_ocr or None
    )
    total_pages = total_rendered_pages or num_pages
    for idx, ocr_txt in ocr_pages_map.items():
        if ocr_txt.strip():
            page_texts[idx] = ocr_txt

    ordered_all = [
        page_texts[i]
        for i in range(total_pages)
        if i in page_texts and page_texts[i].strip()
    ]
    combined_text = "\n\n".join(ordered_all)

    if not combined_text.strip():
        raise RuntimeError(
            "Không trích xuất được văn bản từ tệp PDF này. "
            "Hãy kiểm tra cài đặt `easyocr` hoặc Tesseract OCR Tiếng Việt."
        )

    total_sec = round(time.perf_counter() - started_all, 2)
    print(
        f"[+] Hoàn tất OCR {len(ocr_pages_map)}/{total_pages} trang bằng [{ocr_engine_label}] trong {total_sec} giây!"
    )

    return {
        "text": normalize_vietnamese_text(combined_text),
        "extraction_method": f"Tự động OCR ({ocr_engine_label} - {len(ocr_pages_map)}/{total_pages} trang trong {total_sec}s)",
        "pages": total_pages,
        "ocr_used": True,
        "hardware_used": ocr_engine_label,
    }


def _ocr_pdf_pages_accelerated(
    file_bytes: bytes,
    target_page_indices: list[int] | None = None,
) -> tuple[dict[int, str], str, int]:
    """
    Kết xuất trang PDF ở độ phân giải sắc nét (`scale=2.0` ~ 144 DPI), khử con dấu đỏ,
    và chạy OCR Tiếng Việt với thuật toán tái tạo dòng theo tọa độ trục Y (`detail=1, paragraph=False`).
    """
    if not HAS_PDFIUM or pdfium is None:
        return ({}, "Chưa cài pypdfium2", 0)

    t_render = time.perf_counter()
    pdf_doc = pdfium.PdfDocument(file_bytes)
    page_count = len(pdf_doc)
    indices = (
        [i for i in target_page_indices if 0 <= i < page_count]
        if target_page_indices is not None
        else list(range(page_count))
    )

    # Render ở scale=1.95 (~140 DPI) để giữ sắc nét gạch ngang chữ 'đ' và dấu hỏi/ngã tiếng Việt
    raw_pil_pages: dict[int, Image.Image] = {}
    for idx in indices:
        page = pdf_doc[idx]
        bitmap = page.render(scale=1.95)
        raw_pil_pages[idx] = bitmap.to_pil()

    rendered_images: dict[int, Image.Image] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, max(2, os.cpu_count() or 4))
    ) as pool:
        future_map = {
            pool.submit(_remove_red_stamps_and_preprocess, img): idx
            for idx, img in raw_pil_pages.items()
        }
        for fut in concurrent.futures.as_completed(future_map):
            idx = future_map[fut]
            rendered_images[idx] = fut.result()

    print(
        f"    -> Đã kết xuất (140 DPI) & khử dấu đỏ {len(rendered_images)} trang PDF trong {round((time.perf_counter() - t_render)*1000)} ms."
    )

    results_map: dict[int, str] = {}

    hw = detect_hardware_accelerator()
    has_tess_vie = _configure_windows_tesseract_voice_vie()
    if has_tess_vie and (
        not hw["gpu_available"] or os.environ.get("PREFER_TESSERACT_VIE") == "1"
    ):
        def _tesseract_one_page(item: tuple[int, Image.Image]) -> tuple[int, str]:
            p_idx, p_img = item
            txt = pytesseract.image_to_string(
                p_img, lang="vie", config="--oem 1 --psm 6"
            )
            return (p_idx, txt)

        max_workers = min(8, max(2, os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for p_idx, txt in pool.map(
                _tesseract_one_page, sorted(rendered_images.items())
            ):
                if txt.strip():
                    results_map[p_idx] = txt
        if results_map:
            return (
                results_map,
                f"Tesseract-OCR Tiếng Việt (vie - {max_workers} luồng song song)",
                page_count,
            )

    # ƯU TIÊN SỐ 1 TRÊN GPU RTX 4050: EasyOCR Tiếng Việt với detail=1, paragraph=False
    # để tái tạo chính xác từng dòng văn bản theo tọa độ Y (không bao giờ làm gộp Điều vào Tiêu đề)
    if HAS_EASYOCR and easyocr is not None:
        try:
            reader, accel_label, is_gpu = _get_easyocr_reader()
            ocr_batch_size = 16 if is_gpu else 4
            for step_no, idx in enumerate(sorted(rendered_images.keys()), start=1):
                t_page = time.perf_counter()
                img_np = np.asarray(rendered_images[idx])
                raw_boxes = reader.readtext(
                    img_np,
                    detail=1,
                    paragraph=False,
                    batch_size=ocr_batch_size,
                    workers=0,
                    canvas_size=1600,
                    mag_ratio=1.15,
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                )
                page_str = _reconstruct_lines_from_ocr_boxes(raw_boxes)
                if page_str.strip():
                    results_map[idx] = page_str
                elapsed_p = round((time.perf_counter() - t_page) * 1000)
                if step_no == 1 or step_no % 5 == 0 or step_no == len(rendered_images):
                    print(
                        f"    + [{accel_label}] Đã OCR trang {step_no}/{len(rendered_images)} ({elapsed_p} ms/trang)"
                    )
            if results_map:
                return (results_map, accel_label, page_count)
        except Exception as exc:
            print(f"[!] Lỗi khi chạy EasyOCR: {exc}")

    if has_tess_vie:
        def _tesseract_fallback(item: tuple[int, Image.Image]) -> tuple[int, str]:
            p_idx, p_img = item
            txt = pytesseract.image_to_string(
                p_img, lang="vie", config="--oem 1 --psm 6"
            )
            return (p_idx, txt)

        max_workers = min(8, max(2, os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for p_idx, txt in pool.map(
                _tesseract_fallback, sorted(rendered_images.items())
            ):
                if txt.strip():
                    results_map[p_idx] = txt
        if results_map:
            return (
                results_map,
                f"Tesseract-OCR Tiếng Việt (vie - {max_workers} luồng)",
                page_count,
            )

    if HAS_RAPIDOCR and RapidOCR is not None:
        rapid_engine, rapid_label = _get_rapidocr_engine()
        for idx in sorted(rendered_images.keys()):
            rgb_img = rendered_images[idx].convert("RGB")
            img_array = np.asarray(rgb_img)
            result, _ = rapid_engine(img_array)
            if result:
                lines = [str(item[1]) for item in result if len(item) > 1]
                results_map[idx] = "\n".join(lines)
        if results_map:
            return (results_map, rapid_label, page_count)

    return ({}, "Chưa có Engine OCR Tiếng Việt khả dụng", page_count)


ALLOWED_UPLOAD_EXTENSIONS: set[str] = {".doc", ".docx", ".pdf", ".txt"}

def validate_uploaded_file_format(filename: str, file_bytes: bytes | None = None) -> tuple[bool, str, str]:
    """Kiểm tra định dạng tệp tải lên (phần mở rộng và chữ ký tệp cơ bản)."""
    clean_name = (filename or "").strip()
    ext = Path(clean_name).suffix.lower()
    if not ext:
        return (
            False,
            ext,
            f"Tệp '{clean_name}' không có phần mở rộng định dạng. Hệ thống chỉ hỗ trợ tệp .doc, .docx, .pdf và .txt.",
        )
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return (
            False,
            ext,
            f"Định dạng tệp '{ext}' (tệp: {clean_name}) không phù hợp! Vui lòng chỉ chọn tệp văn bản thuộc các định dạng: .doc, .docx, .pdf hoặc .txt.",
        )
    if file_bytes is not None:
        if len(file_bytes) == 0:
            return (False, ext, f"Tệp '{clean_name}' bị rỗng (0 byte). Vui lòng chọn tệp hợp lệ.")
        if ext == ".pdf" and not file_bytes[:1024].lstrip().startswith(b"%PDF"):
            return (
                False,
                ext,
                f"Tệp '{clean_name}' có đuôi .pdf nhưng cấu trúc bên trong không phải tệp PDF hợp lệ.",
            )
        if ext == ".docx" and not file_bytes[:4].startswith(b"PK\x03\x04"):
            return (
                False,
                ext,
                f"Tệp '{clean_name}' có đuôi .docx nhưng cấu trúc bên trong không phải tệp Word (.docx) hợp lệ.",
            )
    return (True, ext, "Định dạng tệp hợp lệ.")


def extract_text_from_uploaded_file(
    filename: str,
    file_bytes: bytes,
    force_ocr: bool = False,
) -> dict[str, Any]:
    """Hàm điều phối trích xuất văn bản tự động hỗ trợ .doc, .docx, .pdf (số & scan) và .txt."""
    global _LAST_USED_OCR_ENGINE_LABEL
    is_valid_fmt, ext, fmt_msg = validate_uploaded_file_format(filename, file_bytes)
    if not is_valid_fmt:
        raise ValueError(fmt_msg)

    if ext == ".docx":
        return extract_text_from_docx(file_bytes)
    if ext == ".doc":
        return extract_text_from_doc(file_bytes)
    if ext == ".pdf":
        res = extract_text_from_pdf(file_bytes, force_ocr=force_ocr)
        method_str = str(res.get("extraction_method", ""))
        if "EasyOCR" in method_str:
            _LAST_USED_OCR_ENGINE_LABEL = "Easy OCR Tiếng Việt có dấu"
        elif "Tesseract" in method_str:
            _LAST_USED_OCR_ENGINE_LABEL = "Tesseract OCR Tiếng Việt có dấu"
        elif "RapidOCR" in method_str:
            _LAST_USED_OCR_ENGINE_LABEL = "RapidOCR (Chưa có gói Tiếng Việt có dấu)"
        return res
    if ext == ".txt":
        raw = file_bytes.decode("utf-8", errors="replace")
        return {
            "text": normalize_vietnamese_text(raw),
            "extraction_method": f"Văn bản thuần ({ext})",
            "pages": 1,
            "ocr_used": False,
            "hardware_used": "CPU",
        }
    raise ValueError(fmt_msg)


def validate_legal_document(
    text: str,
    filename: str = "",
    *,
    use_llm_check: bool = True,
    min_score: int = 35,
) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if len(cleaned) < 80:
        return {
            "is_legal_document": False,
            "confidence_score": 0,
            "detected_doc_number": "CHƯA_RÕ_SỐ_HIỆU",
            "detected_doc_type": "Văn bản pháp quy khác",
            "detected_title": filename or "Văn bản không xác định",
            "article_count": 0,
            "matched_signals": [],
            "missing_signals": ["Nội dung văn bản quá ngắn (< 80 ký tự)"],
            "diacritics_warning": False,
            "llm_verification": None,
            "reason": "Văn bản quá ngắn hoặc không trích xuất được chữ.",
        }

    score = 0
    matched_signals: list[str] = []
    missing_signals: list[str] = []

    for label, pattern, weight in LEGAL_PATTERNS:
        if pattern.search(cleaned):
            score += weight
            matched_signals.append(label)
        else:
            missing_signals.append(label)

    article_matches = ARTICLE_SPLIT_RE.findall(cleaned)
    if len(article_matches) >= 2:
        score += 15
        matched_signals.append(
            f"Có hệ thống điều khoản chuẩn ({len(article_matches)} Điều)"
        )

    score = min(100, score)

    detected_doc_number = _extract_best_doc_number(cleaned, filename=filename)
    detected_title = _infer_document_title(cleaned, filename, detected_doc_number)
    detected_doc_type = classify_legal_doc_type(
        detected_doc_number, detected_title, cleaned[:1500]
    )

    missing_diacritics = _has_missing_vietnamese_diacritics(cleaned)

    llm_result: dict[str, Any] | None = None
    if use_llm_check and score < 60:
        llm_result = _verify_legal_with_ollama(cleaned[:1200])
        if llm_result and llm_result.get("checked"):
            if llm_result.get("is_legal") is True:
                score = max(score, 75)
            elif llm_result.get("is_legal") is False and score < 60:
                score = min(score, 25)

    is_valid = score >= min_score
    reason = (
        f"Đạt chuẩn {detected_doc_type} (Điểm nhận diện: {score}/100; "
        f"Khớp {len(matched_signals)} dấu hiệu pháp quy; Tách được {len(article_matches)} Điều)."
        if is_valid
        else f"Cảnh báo: Tệp tải lên có ít dấu hiệu của văn bản quy phạm pháp luật/quy chế "
        f"(Điểm nhận diện: {score}/100 < ngưỡng {min_score})."
    )
    if missing_diacritics:
        reason += (
            " ⚠️ [CẢNH BÁO MẤT DẤU TIẾNG VIỆT: Văn bản trích xuất đang thiếu dấu Tiếng Việt!]"
        )

    return {
        "is_legal_document": is_valid,
        "confidence_score": score,
        "detected_doc_number": detected_doc_number,
        "detected_doc_type": detected_doc_type,
        "detected_title": detected_title,
        "article_count": len(article_matches),
        "matched_signals": matched_signals,
        "missing_signals": missing_signals[:4],
        "diacritics_warning": missing_diacritics,
        "llm_verification": llm_result,
        "reason": reason,
    }


def _infer_document_title(text: str, filename: str, doc_number: str) -> str:
    """
    Tự động suy luận tên trích yếu của văn bản từ phần mở đầu:
    - Cắt bỏ hoàn toàn đoạn 'Căn cứ...', 'Theo đề nghị...', và tạp chất con dấu.
    - Giới hạn độ dài tên trích yếu <= 180 ký tự để luôn hiển thị gọn đẹp.
    """
    cleaned_header = _post_correct_vietnamese_legal_ocr(text[:2500])
    lines = [ln.strip() for ln in cleaned_header.splitlines() if ln.strip()][:35]
    doc_type_re = re.compile(
        r"^(LUẬT|BỘ LUẬT|NGHỊ ĐỊNH|NGHI DINH|THÔNG TƯ|THONG TU|QUYẾT ĐỊNH|QUYET DINH|QUY CHẾ|QUY ĐỊNH|NGHỊ QUYẾT|CHỈ THỊ|HƯỚNG DẪN)\b",
        re.IGNORECASE,
    )
    stop_words_re = re.compile(
        r"\b(Căn\s+cứ|Can\s+cu|Theo\s+đề\s+nghị|Chương\s+[IVX\d]+|Điều\s+1\b|Chính\s+phủ\s+ban\s+hành|Bộ\s+trưởng\s+ban\s+hành|Quốc\s+hội\s+ban\s+hành)",
        re.IGNORECASE,
    )

    for idx, line in enumerate(lines):
        if doc_type_re.match(line):
            parts = [line]
            for follow_idx in range(idx + 1, min(idx + 4, len(lines))):
                nxt = lines[follow_idx]
                if stop_words_re.search(nxt):
                    nxt_cut = stop_words_re.split(nxt)[0].strip()
                    if nxt_cut:
                        parts.append(nxt_cut)
                    break
                parts.append(nxt)
                if len(" ".join(parts)) >= 140:
                    break

            combined_title = " ".join(parts)
            # Cắt bỏ phần 'Căn cứ...' nếu bị dính trên cùng dòng đầu
            if stop_words_re.search(combined_title):
                combined_title = stop_words_re.split(combined_title)[0].strip()
            combined_title = re.sub(r"\s+", " ", combined_title).strip(" ,;-|")
            if len(combined_title) > 180:
                combined_title = combined_title[:177].rsplit(" ", 1)[0] + "..."
            return combined_title

    stem = Path(filename).stem if filename else "Văn bản pháp luật"
    return f"{stem} ({doc_number})"


def _clean_article_header_line(raw_header: str) -> str:
    """
    Làm sạch dòng tiêu đề Điều (`Điều X. Tên điều`):
    - Nếu nội dung của Khoản 1 (`1. Phạt tiền...`) bị dính vào cùng dòng với tên Điều,
      cắt bỏ phần từ `1. ` trở đi khỏi tiêu đề Điều để tiêu đề thẻ kết quả luôn ngắn gọn!
    """
    header = re.sub(r"\s+", " ", (raw_header or "").strip())
    # Cắt tại vị trí bắt đầu Khoản 1 (' 1. ') nếu nằm sau tên Điều
    clause_1_match = re.search(r"(?<=\w)\s+1\.\s+[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆ]", header)
    if clause_1_match and clause_1_match.start() >= 10:
        header = header[: clause_1_match.start()].strip()
    if len(header) > 165:
        header = header[:162].rsplit(" ", 1)[0] + "..."
    return header


def _verify_legal_with_ollama(excerpt: str) -> dict[str, Any]:
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "qwen2.5:7b")
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Bạn là bộ kiểm duyệt văn bản pháp luật Việt Nam và quy chế đào tạo. "
                            "Chỉ trả về JSON hợp lệ theo mẫu: "
                            '{"is_legal": true/false, "category": "Luật/Nghị định/Thông tư/Quyết định/Quy chế"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Đoạn trích đầu văn bản:\n{excerpt}",
                    },
                ],
            },
            timeout=3.5,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return {
                "checked": True,
                "is_legal": bool(parsed.get("is_legal", True)),
                "category": str(parsed.get("category", "")),
                "model": model,
            }
    except Exception:
        pass
    return {"checked": False, "is_legal": None, "category": None, "model": model}


def chunk_legal_document(
    text: str,
    *,
    doc_number: str,
    doc_title: str,
    source: str,
    doc_type: str | None = None,
    max_chars: int = 2200,
    id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_vietnamese_text(text)
    if not normalized:
        return []

    body_text = strip_legal_boilerplate(normalized)
    if not body_text:
        body_text = normalized

    clean_doc_num = (doc_number or "VB-MOI").strip()
    short_doc_title = _infer_document_title(doc_title, "", clean_doc_num)
    resolved_doc_type = doc_type or classify_legal_doc_type(
        clean_doc_num, short_doc_title, normalized[:800]
    )
    safe_prefix = re.sub(r"[^A-Z0-9]+", "_", clean_doc_num.upper().replace("Đ", "D")).strip("_") or "VB"
    prefix = id_prefix or safe_prefix
    now_iso = dt.datetime.now().isoformat(timespec="seconds")

    matches = list(ARTICLE_SPLIT_RE.finditer(body_text))
    chunks: list[dict[str, Any]] = []

    if matches:
        for idx, match in enumerate(matches):
            start_pos = match.start()
            end_pos = (
                matches[idx + 1].start() if idx + 1 < len(matches) else len(body_text)
            )
            article_block = body_text[start_pos:end_pos].strip()
            raw_article_header = match.group(1).strip()
            clean_header = _clean_article_header_line(raw_article_header)

            # Trích xuất số Điều thực tế (VD: 'Điều 9.' -> 9) để đặt mã ID chính xác (_D9 thay vì _D2)
            art_num_match = re.search(r"Điều\s+(\d+[a-z]?)", clean_header, re.IGNORECASE)
            art_num_str = art_num_match.group(1) if art_num_match else str(idx + 1)

            preceding_text = body_text[:start_pos]
            chapter_matches = list(CHAPTER_HEADING_RE.finditer(preceding_text))
            chapter_ctx = (
                f" [{_clean_article_header_line(chapter_matches[-1].group(1))}]"
                if chapter_matches
                else ""
            )

            trailing_chapter = CHAPTER_HEADING_RE.search(article_block)
            if trailing_chapter and trailing_chapter.start() > len(raw_article_header):
                article_block = article_block[: trailing_chapter.start()].strip()

            sub_blocks = _split_long_article(
                article_block, clean_header=clean_header, max_chars=max_chars
            )
            for part_idx, sub_text in enumerate(sub_blocks, start=1):
                part_suffix = f" (Phần {part_idx})" if len(sub_blocks) > 1 else ""
                chunk_id = f"{prefix}_D{art_num_str}" + (
                    f"_P{part_idx}" if len(sub_blocks) > 1 else ""
                )
                chunk_title = (
                    f"[{clean_doc_num}] {short_doc_title}{chapter_ctx} - {clean_header}{part_suffix}"
                )
                chunks.append(
                    {
                        "id": chunk_id,
                        "title": chunk_title,
                        "text": sub_text,
                        "source": source,
                        "doc_number": clean_doc_num,
                        "doc_type": resolved_doc_type,
                        "status": "active",
                        "added_at": now_iso,
                    }
                )
        return chunks

    paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    buffer = ""
    part_no = 1
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para}".strip()
        else:
            if buffer:
                chunks.append(
                    {
                        "id": f"{prefix}_M{part_no}",
                        "title": f"[{clean_doc_num}] {short_doc_title} - Mục/Đoạn {part_no}",
                        "text": buffer,
                        "source": source,
                        "doc_number": clean_doc_num,
                        "doc_type": resolved_doc_type,
                        "status": "active",
                        "added_at": now_iso,
                    }
                )
                part_no += 1
            buffer = para

    if buffer:
        chunks.append(
            {
                "id": f"{prefix}_M{part_no}",
                "title": f"[{clean_doc_num}] {short_doc_title} - Mục/Đoạn {part_no}",
                "text": buffer,
                "source": source,
                "doc_number": clean_doc_num,
                "doc_type": resolved_doc_type,
                "status": "active",
                "added_at": now_iso,
            }
        )

    return chunks


def _split_long_article(
    article_text: str, clean_header: str = "", max_chars: int = 2200
) -> list[str]:
    if len(article_text) <= max_chars:
        return [article_text]

    lines = article_text.splitlines()
    header_line = clean_header or lines[0].strip()

    # Nếu dòng đầu tiên chứa cả tiêu đề Điều lẫn nội dung Khoản 1 bị dính, tách phần nội dung Khoản 1 ra
    first_line_rest = lines[0].strip()
    if clean_header and first_line_rest.startswith(clean_header):
        remainder = first_line_rest[len(clean_header) :].strip()
        body_lines = ([remainder] if remainder else []) + lines[1:]
    else:
        body_lines = lines[1:]

    # Nếu toàn bộ Điều bị dính trên 1 dòng dài, chia theo dấu chấm phẩy '; ' hoặc câu '. '
    expanded_lines: list[str] = []
    for ln in body_lines:
        if len(ln) > max_chars * 0.8:
            sub_clauses = re.split(r"(?<=[\;\.])\s+(?=[a-zđ\d]{1,2}[\)\.])", ln)
            expanded_lines.extend(sub_clauses)
        else:
            expanded_lines.append(ln)

    parts: list[str] = []
    current_lines: list[str] = [header_line]
    current_len = len(header_line)

    for line in expanded_lines:
        if current_len + len(line) + 1 > max_chars and len(current_lines) > 1:
            parts.append("\n".join(current_lines).strip())
            current_lines = [f"{header_line} (tiếp theo)", line]
            current_len = len(current_lines[0]) + len(line)
        else:
            current_lines.append(line)
            current_len += len(line) + 1

    if len(current_lines) > 1:
        parts.append("\n".join(current_lines).strip())

    return parts or [article_text]