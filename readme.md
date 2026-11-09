# Hệ Thống AI Tra Cứu Văn Bản Pháp Luật và Quy Chế Đào Tạo Thông Minh

Hệ thống tra cứu, số hóa và quản trị văn bản quy phạm pháp luật, quy chế đào tạo thông minh ứng dụng thuật toán **Tìm kiếm Lai (Hybrid Search: Okapi BM25 + Dense Retrieval `BAAI/bge-m3` + Reciprocal Rank Fusion `RRF`)**, kết hợp công nghệ **Nhận dạng ký tự quang học (`EasyOCR` khử con dấu đỏ trên GPU)** và **Mô hình Ngôn ngữ Lớn Cục bộ (`Local LLM Qwen 2.5 7B` qua Ollama)**.

> **Công trình Sáng kiến Chuyển đổi số, Khoa học Công nghệ và Đổi mới sáng tạo — Học viện An ninh nhân dân (T01), 2026**  
> **Định hướng ứng dụng:** Phục vụ công tác giảng dạy, học tập, nghiên cứu khoa học và quản lý đào tạo trong môi trường mạng nội bộ (`100% Offline / Air-Gapped Intranet`).

## Kết Quả Thực Nghiệm Nổi Bật (Key Results)

| Chỉ số đánh giá (Metric) | Mô hình gốc (Baseline) | Hệ thống nâng cấp (Sáng kiến) | Mức độ cải thiện / Ý nghĩa thực tiễn |
| :--- | :---: | :---: | :--- |
| **BM25 `Recall@1`** | `66,11%` (Regex `\w+`) | **`72,78%`** (Tách từ ghép `pyvi`) | Tăng **`+6,67%`** nhờ bảo toàn từ ghép pháp lý tiếng Việt |
| **BM25 `MRR@10`** | `88,06%` | **`93,33%`** | Tăng **`+5,27%`** thứ hạng xuất hiện điều khoản đúng đầu tiên |
| **Dense Retrieval `Recall@10`** | `98,33%` (`BAAI/bge-m3`) | **`98,33%`** (`FP16 CUDA`) | Độ bao phủ ngữ nghĩa đa ngôn ngữ 1.024 chiều |
| **Hybrid RRF `Recall@5`** | `92,22%` | **`95,56%`** | Tăng **`+3,34%`** độ bao phủ trong 5 kết quả đầu tiên |
| **Hybrid RRF `MRR@10`** | `97,08%` | **`97,78%`** | **29/30** truy vấn chuẩn có điều khoản đúng nằm ngay **Hạng #1** |
| **Hybrid RRF `nDCG@10`** | `94,83%` | **`95,34%`** | Tối ưu hóa thứ tự xếp hạng các điều khoản đa mức độ |
| **Thời gian nạp chỉ mục khởi động** | `118,50 giây` (Cold build) | **`0,18 giây`** (`embeddings.npy`) | Nhanh gấp **`~650 lần`** nhờ cơ chế lưu đệm ma trận nhị phân |
| **Độ trễ truy vấn (`p50` Latency)** | `204,29 ms` (CPU) | **`21,15 ms`** (`RTX 4050 FP16`) | Phản hồi tức thời dưới `0,03 giây` trên nhân Tensor GPU |

## Kiến Trúc Tổng Thể Hệ Thống (System Architecture)

```text
   ĐẦU VÀO VĂN BẢN ĐA ĐỊNH DẠNG (.doc OLE2, .docx, .pdf số, .pdf scan)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        Văn bản số (.doc/.docx/.pdf)      PDF Scan bị làm phẳng
         Bóc tách trực tiếp Unicode              │
                    │               Tiền xử lý RGB khử con dấu đỏ
                    │               + EasyOCR Tiếng Việt (GPU CUDA)
                    │               + Ghép dòng theo tọa độ trục Y
                    └──────────────┬──────────────┘
                                   ▼
            Lọc Quốc hiệu/Header + Hậu kiểm chính tả pháp lý
            & Cắt phân đoạn bảo toàn ngữ cảnh (Chương - Điều - Khoản)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         CHỈ MỤC TỪ VỰNG (BM25)        CHỈ MỤC NGỮ NGHĨA (DENSE)
        Tách từ ghép Tiếng Việt       Mô hình BAAI/bge-m3 (1024-d)
       (pyvi + bm25_tokens_cache)     (Lưu đệm nhị phân embeddings.npy)
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                 TẦNG DUNG HỢP THỨ HẠNG NGHỊCH ĐẢO (RRF)
                 Score_RRF(d) = Σ w_i / (k + rank_i(d))
                                   ▲
                                   │
                 MÔ-ĐUN CHUẨN HÓA CÂU HỎI PHÁP LÝ 2 TẦNG
          Tầng 1: Từ điển ánh xạ khẩu ngữ/viết tắt -> Thuật ngữ pháp điển
          Tầng 2: Mô hình ngôn ngữ lớn cục bộ Qwen 2.5 (7B) qua Ollama
                                   │
                                   ▼
             GIAO DIỆN WEB TRỰC QUAN & RESTFUL API (FastAPI)
     ┌─────────────────────────────┼─────────────────────────────┐
     ▼                             ▼                             ▼
[Tab 1: Tra cứu thông minh]  [Tab 2: Quản lý văn bản]  [Tab 3: Nhật ký & Đánh giá]
```

## Các Tính Năng Cốt Lõi (Features)

- **Tìm kiếm Lai (Hybrid Search) độ chính xác cao** — Kết hợp đồng thời thuật toán xác suất từ vựng `Okapi BM25` (đã tối ưu bộ tách từ ghép tiếng Việt `pyvi`) và tìm kiếm vector ngữ nghĩa `BAAI/bge-m3` thông qua thuật toán dung hợp thứ hạng `Reciprocal Rank Fusion (RRF)`.
- **Kiến trúc Chuẩn hóa Câu hỏi Pháp lý 2 Tầng** — Tự động dịch các câu hỏi khẩu ngữ đời thường, từ viết tắt (`vphc`, `gplx`, `anm`...) và đại từ tham chiếu hội thoại nhiều lượt sang thuật ngữ pháp điển chính quy bằng quy tắc Heuristic (`< 1 ms`) kết hợp `Local LLM Qwen 2.5 (7B)`.
- **Tự động sinh Câu hỏi Gợi ý theo kho văn bản** — Tự động trích xuất các chủ đề thực chất từ các văn bản mới nạp vào kho để tạo danh sách thẻ gợi ý tra cứu nhanh trên giao diện (tự động loại bỏ số hiệu rườm rà và các điều khoản thủ tục cuối văn bản).
- **Số hóa đa định dạng & OCR Tiếng Việt khử con dấu đỏ trên GPU** — Đọc trực tiếp tệp Word 97–2003 (`.doc` nhị phân `OLE2`), Word mới (`.docx`), `.txt`, `.pdf` lớp văn bản và `.pdf` bản quét (scan). Tích hợp thuật toán tách kênh màu `RGB` loại bỏ con dấu đỏ đè lên chữ đen, ghép dòng theo tọa độ `Y`, lọc bỏ Quốc hiệu/Tiêu ngữ và tự động sửa lỗi chính tả pháp lý sau OCR.
- **Cửa sổ Xem trước Tài liệu Gốc (Windows Explorer Preview Pane)** — Cho phép đối chiếu song song (`Split View`) giữa danh sách các điều khoản đã bóc tách và tài liệu gốc (hỗ trợ xem trực tiếp trang PDF gốc hoặc chế độ trang giấy A4 tự động tô sáng `Heading Chương/Mục` và `Heading Điều`).
- **Quản trị vòng đời văn bản & Cảnh báo trùng lặp thông minh** — Tự động nhận diện Số hiệu, phân loại vào 05 nhóm (*Luật / Bộ luật*, *Nghị định*, *Thông tư*, *Quyết định*, *Quy chế / Quy định*), phát hiện trùng số hiệu hoặc trùng nội dung điều khoản, và cho phép bấm trực tiếp vào nhãn **`Hiệu lực`** để đánh dấu **Hết hiệu lực** cho từng điều khoản riêng lẻ hoặc toàn bộ văn bản trong cơ sở dữ liệu.
- **Biểu đồ đánh giá trực quan & 05 lớp bảo mật nội bộ** — Tích hợp biểu đồ cột SVG đối sánh hiệu năng ngay tại Tab 3; bảo đảm an toàn thông tin nhờ kiến trúc `100% Offline (Air-Gapped)`, kiểm tra chữ ký tệp nhị phân (`Magic-Byte`), chống `XSS`, khóa luồng `RLock` và nhật ký kiểm toán `Audit Logs`.

## Môi Trường Thử Nghiệm & Tương Thích Phần Cứng (Lab Environment)

| Thành phần | Thông số kỹ thuật trên máy kiểm thử | Cơ chế tự động thích ứng đa nền tảng |
| :--- | :--- | :--- |
| **CPU** | AMD Ryzen 7 (8 Nhân / 16 Luồng xử lý) | Tự động chạy đa luồng (`CPU Multi-threading`) trên mọi dòng máy văn phòng |
| **GPU rời (dGPU)** | NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) | Tự động kích hoạt nhân `CUDA 12.1` & tính toán bán phần `FP16 Tensor Cores` |
| **GPU tích hợp (iGPU)** | AMD Radeon / Intel Iris Xe Graphics | Hỗ trợ tăng tốc qua `torch-directml` trên các dòng laptop mỏng nhẹ |
| **RAM & Lưu trữ** | 16 GB DDR5 RAM, NVMe SSD | Nạp ma trận nhị phân `embeddings.npy` vào RAM chỉ trong `0,18 giây` |
| **Mô hình AI cục bộ** | `BAAI/bge-m3` + `EasyOCR` + `Ollama qwen2.5:7b` | Vận hành khép kín tại `localhost` (`127.0.0.1`), không phụ thuộc Internet |

## Hướng Dẫn Cài Đặt (Installation)

### Yêu cầu tiên quyết (Prerequisites)

- Hệ điều hành **Windows 10 / 11 (64-bit)** hoặc **Linux (Ubuntu 20.04+)**
- **Python 3.10 – 3.11**
- *(Tùy chọn)* **Ollama** cài đặt trên máy với mô hình `qwen2.5:7b` để kích hoạt chuẩn hóa câu hỏi Tầng 2

### 1. Tải kho mã nguồn (Clone repository)

```powershell
git clone https://github.com/YOUR_USERNAME/he-thong-tra-cuu-phap-luat-ai-t01.git
cd he-thong-tra-cuu-phap-luat-ai-t01
```

### 2. Khởi tạo môi trường ảo Python (`.venv`)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Cài đặt các gói thư viện phụ thuộc

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Kích hoạt tăng tốc phần cứng GPU NVIDIA (Dành cho máy có Card rời NVIDIA RTX/GTX)

Nếu máy tính có card đồ họa rời NVIDIA, chạy lệnh tự động cài đặt bản `PyTorch CUDA 12.1` tương thích:

```powershell
python hybrid_search.py --fix-gpu
```

*(Hoặc cài thủ công: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121`)*

### 5. Cấu hình Ollama & Mô hình Ngôn ngữ Lớn cục bộ (Tùy chọn)

```powershell
# Tải mô hình Qwen 2.5 (7B) về máy cục bộ:
ollama pull qwen2.5:7b
```

*(Lưu ý: Ngay cả khi máy không bật Ollama, hệ thống vẫn hoạt động đầy đủ 100% các chức năng nhờ bộ Chuẩn hóa câu hỏi Tầng 1 và lõi Hybrid Search RRF).*

## Hướng Dẫn Sử Dụng (Usage)

### 1. Khởi chạy Giao diện Web trực quan

```powershell
.\.venv\Scripts\activate
python app.py
```

Kết quả hiển thị trên cửa sổ dòng lệnh:

```text
[*] Đang khởi tạo HybridRetriever từ: corpus.jsonl ...
[1/2] Đang nạp mô hình BAAI/bge-m3 trên thiết bị [CUDA]...
      -> Đã nạp xong mô hình sau 4050 ms.
[2/2] Đang đồng bộ hóa chỉ mục BM25 và Vector Embeddings...
[+] Phần cứng tính toán: NVIDIA CUDA GPU - NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM) (Batch size: 32)
[+] Hoàn tất đồng bộ chỉ mục: 1009 điều khoản (Tái sử dụng 1009 vector từ embeddings.npy, thời gian đồng bộ: 211.33 ms).
[+] Đang khởi chạy máy chủ FastAPI + Uvicorn tại http://127.0.0.1:8000 (Tài liệu API: http://127.0.0.1:8000/docs)
```

Trình duyệt sẽ tự động mở tại địa chỉ `http://127.0.0.1:8000`.

### 2. Tạo tệp thực thi khởi chạy nhanh `TraCuuPhapLuat.exe` (Siêu nhẹ ~24 KB)

Để tạo tệp `TraCuuPhapLuat.exe` gắn biểu tượng Học viện An ninh nhân dân ngay tại thư mục gốc (giúp khởi động bằng 1 cú nhấp đúp chuột mà không sinh ra các thư mục nặng hàng GB):

```powershell
python build_exe.py
```

### 3. Chạy kiểm thử tự động và Đánh giá định lượng (CLI Benchmark)

```powershell
# Kiểm tra toàn vẹn thuật toán RRF và các trường hợp biên:
python hybrid_search.py --self-check

# Chạy đánh giá định lượng trên bộ 30 câu hỏi chuẩn và xuất evaluation/metrics.json:
python hybrid_search.py --benchmark
```

## Quy Tắc Kiểm Duyệt & Xử Lý Trùng Lặp Văn Bản (Ingestion & Deduplication Rules)

| Cấp độ phát hiện | Điều kiện kích hoạt | Hành động khuyến nghị | Mô tả cơ chế xử lý |
| :--- | :--- | :--- | :--- |
| **`exact_duplicate`** | Trùng Số hiệu văn bản + 100% điều khoản trùng mã băm SHA-256 | `replace_existing` | Cảnh báo trùng hoàn toàn; cho phép cập nhật ghi đè hoặc hủy bỏ |
| **`same_doc_number`** | Trùng Số hiệu văn bản + Có điều khoản mới/sửa đổi bổ sung | `replace_existing` | Đánh dấu các điều khoản cũ sang trạng thái `replaced` và nạp bản mới |
| **`partial_content_duplicate`** | Khác Số hiệu nhưng có điều khoản trùng nội dung | `append_new_only` | Chỉ nạp bổ sung các điều khoản mới chưa trùng nội dung |
| **`none` (Văn bản mới)** | Không trùng Số hiệu và không trùng nội dung | `force_add` | Nạp toàn bộ điều khoản, cập nhật `embeddings.npy` trong `< 1 giây` |

## Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
.
├── app.py                        # Máy chủ Web FastAPI & Giao diện 3 phân hệ (Tra cứu | Quản lý | Đánh giá)
├── hybrid_search.py              # Lõi Hybrid Search (BM25 pyvi + Dense BAAI/bge-m3 + RRF + Qwen 2.5 + GPU Fix)
├── document_ingestor.py          # Mô-đun đọc đa định dạng (.doc, .docx, .pdf) & OCR khử dấu đỏ trên GPU
├── build_exe.py                  # Công cụ biên dịch tệp khởi chạy gọn nhẹ TraCuuPhapLuat.exe (~24 KB)
├── prepare_subset.py             # Công cụ chuẩn bị dữ liệu kiểm thử từ tập YuITC Legal Documents
├── requirements.txt              # Danh mục thư viện Python phụ thuộc
├── Logo học viện.jpg             # Huy hiệu Học viện An ninh nhân dân (T01)
├── corpus.jsonl                  # Cơ sở dữ liệu các điều khoản pháp luật & quy chế đào tạo
├── queries.jsonl                 # Tập 30 câu hỏi pháp luật chuẩn phục vụ đánh giá Benchmark
├── embeddings.npy                # Ma trận vector nhúng 1024 chiều lưu đệm nhị phân
├── embeddings_meta.json          # Siêu dữ liệu kiểm tra đồng bộ tự động giữa Corpus và Vector Cache
├── bm25_tokens_cache.json        # Bộ nhớ đệm từ ghép Tiếng Việt đã tách qua pyvi
├── evaluation/
│   ├── metrics.json              # Báo cáo chỉ số Recall@1, Recall@5, Recall@10, MRR@10, nDCG@10, Latency
│   └── per_query.csv             # Kết quả thực nghiệm chi tiết trên từng câu hỏi kiểm thử
└── logs/
    └── query_rewrite_audit.jsonl # Nhật ký kiểm toán lịch sử chuẩn hóa câu hỏi pháp lý & hội thoại
```

## Tài Liệu Tham Khảo (References)

1. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP 2019.
3. Chen, J., Xiao, S., Zhang, P., et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. BAAI (`BAAI/bge-m3`).
4. Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. ACM SIGIR 2009.
5. Lewis, P., Perez, E., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
6. Le, H. P., Nguyen, T. M. H., et al. (2008). *A Hybrid Approach to Vietnamese Word Segmentation* (`pyvi`).
7. Nguyen, L. M., et al. (2021–2024). *Overview of the Automated Legal Question Answering Competition (ALQAC)*. IEEE KSE.
8. YuITC Community (2024). *Vietnamese-Legal-Documents Dataset*. Hugging Face Datasets.

## Giấy Phép & Bản Quyền (License)

Dự án được phát triển phục vụ mục đích giảng dạy, học tập, nghiên cứu khoa học và chuyển đổi số tại **Học viện An ninh nhân dân (T01)**.
