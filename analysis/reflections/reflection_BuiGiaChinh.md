# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Bùi Gia Chính
**MSSV:** 2A202602693
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 2026-10-04

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Trên 26 documents thật (bỏ 2 PDF scan), threshold=0.85 tạo **208 chunks** (avg 99 ký tự, min 6, max 354) — nhiều và nhỏ hơn hẳn basic paragraph chunking (51 chunks, avg 410 ký tự). Semantic chunking nhạy với threshold: 0.85 khá chặt nên tách câu thành nhóm rất nhỏ, có nguy cơ mất context nếu câu liên quan bị tách rời (chunk min=6 ký tự là dấu hiệu over-splitting). |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF giải quyết vấn đề BM25 (khớp từ khóa chính xác, tốt cho số liệu như "12 ký tự", "120 ngày") và Dense (khớp ngữ nghĩa, tốt cho câu hỏi diễn đạt khác từ) không đồng nhất về scale điểm — bằng cách chỉ dùng **rank** thay vì raw score nên không cần normalize. Thực tế: `segment_vietnamese()` phải `replace("_", " ")` sau `underthesea.word_tokenize`, nếu không BM25 tokenize "nghỉ_phép" thành 1 token trong khi query "nghỉ phép" tokenize ra 2 token → không khớp. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Test nhanh với 3 câu (1 liên quan, 2 không liên quan) cho score 0.9914 vs 0.0206 vs 0.0007 — phân biệt rất rõ khi nội dung khác chủ đề. NHƯNG khi 2 chunk gần giống nhau về mặt câu chữ (chính sách mật khẩu v1 "8 ký tự" vs v2 "12 ký tự" — chỉ khác 1 số), reranker cho score gần như bằng nhau (0.998 vs 0.997) — cross-encoder học semantic similarity, không học "văn bản nào còn hiệu lực theo thời gian". Đây là limitation quan trọng phát hiện được từ failure analysis thực tế (xem `analysis/failure_analysis.md` #1, #2). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Chạy thật trên 20 câu hỏi: Production đạt Faithfulness 0.68, Answer Relevancy 0.69, Context Precision 0.97, Context Recall 0.88. Context Precision cao nhất nhờ rerank lọc nhiễu — đúng như lecture dạy reranking tăng precision. Nhưng Faithfulness THẤP HƠN naive baseline (0.68 vs 0.75) — vì pipeline production trả "Không tìm thấy" ở vài câu khi gặp context mâu thuẫn (an toàn nhưng bị RAGAS chấm faithfulness=0, vì không entail được ground truth) trong khi naive baseline (không hybrid, không rerank) vô tình chỉ lấy 1 phiên bản chính sách nên "trả lời bừa" lại đúng hơn theo metric. |
| Contextual embeddings | M5 | `_enrich_single_call()` | Dùng combined mode (1 API call/chunk, không phải 4 call riêng) để tiết kiệm cost — xác nhận qua log thực tế: enrichment cho 107 chunks tốn **574.3s** (52% tổng thời gian pipeline), là bottleneck lớn nhất. Context prepend dạng "Đoạn văn nằm trong phần... của tài liệu..." giúp chunk tự giải thích vị trí của nó, nhưng cũng là nguyên nhân phụ khiến các chunk v1/v2 chính sách mật khẩu bị embedding gần giống nhau hơn (câu enrichment mô tả giống nhau dù nội dung số liệu khác nhau). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `UnicodeEncodeError: 'charmap' codec can't encode character '\u1ecb' in position 28: character maps to <undefined>` khi chạy `pip install -r requirements.txt`, kể cả `pip --version` cũng crash với lỗi tương tự.
  - Sau khi fix lỗi trên, numpy build thành công nhưng in cảnh báo: `Numpy built with MINGW-W64 on Windows 64 bits is experimental... CRASHES ARE TO BE EXPECTED`.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Project nằm ở đường dẫn có ký tự tiếng Việt có dấu (`D:\Xịn\Học hành\VinUni\Lab18\...`). Codepage console/file mặc định của Windows (cp1258) không mã hóa được một số ký tự Unicode tổ hợp sẵn (precomposed) trong đường dẫn, khiến bất kỳ tool nào ghi path ra text (kể cả `pip` nội bộ) bị crash. Debug bằng cách thử `pip --version` trực tiếp để cô lập lỗi về đúng pip/encoding chứ không phải do riêng numpy.
  - Fix tạm: set `PYTHONUTF8=1` để Python dùng UTF-8 mode cho toàn bộ text I/O, giải quyết được crash ngay.
  - Nhưng nguyên nhân sâu hơn: Python 3.13 (bản đang cài) chưa có pre-built wheel cho numpy 1.26.4 (vì `ragas==0.1.22` pin `numpy<2`), nên pip phải build from source — và trên Windows không có MSVC/MKL sẵn nên fallback sang MinGW, bản mà numpy tự cảnh báo là **experimental, có thể crash**. Quyết định: dựng lại `.venv` bằng Python 3.11 (có pre-built wheel numpy 1.26.4 chính thức) để loại bỏ rủi ro này hoàn toàn, thay vì chỉ che lỗi encoding.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Chưa hiểu rõ cơ chế resolver của pip khi nhiều package transitive pin các range version numpy khác nhau (`ragas<0.2` → kéo theo `numpy<2`) → đã học cách trace bằng cách đọc log `pip install` đầy đủ (`Building wheels for collected packages: numpy`) để hiểu vì sao pip không chọn numpy mới hơn.
  - Lần đầu dùng OpenAI-compatible proxy (ShopAIKey) thay vì OpenAI key gốc — tìm hiểu cách SDK `openai` tự đọc `OPENAI_BASE_URL` từ env var mà không cần sửa code, và `langchain-openai`/`ragas` cần thêm `OPENAI_API_BASE` (tên biến legacy) để tương thích.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: [Điền tên project cá nhân của bạn — ví dụ: Chatbot hỗ trợ nội bộ / RAG cho tài liệu công ty]

#### 1. Hiện trạng
- **Pipeline hiện tại:** *(cần điền theo project thật của bạn — ví dụ: RAG đơn giản dùng paragraph chunking + dense-only search, chưa có reranking/evaluation)*
- **Vấn đề / Bottlenecks đang gặp:** *(ví dụ: không phân biệt được tài liệu cũ/mới, retrieval multi-hop yếu, chưa có metric đánh giá định lượng)*

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical (parent 2048 / child 256) — vì cho precision tốt khi retrieve (child nhỏ, match chính xác) mà vẫn giữ context đầy đủ khi trả về (return parent). Semantic chunking (threshold 0.85) quá nhạy, dễ over-split (đã thấy chunk 6 ký tự trong lab) nên chỉ dùng cho tài liệu văn xuôi dài, không dùng cho tài liệu dạng chính sách/quy định có số liệu.
2. **Search retrieval:** Hybrid (BM25 + Dense + RRF) — vì tài liệu có nhiều số liệu/mã chính xác (ngày, %, VNĐ) mà dense-only dễ bỏ sót, còn BM25-only lại kém với câu hỏi diễn đạt lại.
3. **Reranking:** Có — cross-encoder `bge-reranker-v2-m3`, nhưng bổ sung **metadata filtering theo version/ngày hiệu lực TRƯỚC khi rerank**, để tránh lỗi đã gặp trong lab (2 phiên bản chính sách mâu thuẫn lọt cùng top-3).
4. **Evaluation:** RAGAS 4 metrics làm baseline, nhưng chạy evaluate 2-3 lần lấy trung bình vì quan sát thấy judge LLM có variance (case #5 trong failure_analysis: câu trả lời đúng nội dung vẫn bị chấm faithfulness=0).
5. **Enrichment:** Contextual prepend (combined single-call mode) — hiệu quả nhưng là bottleneck latency lớn nhất (52% thời gian pipeline trong lab này); cần parallelize bằng `asyncio`/batch API call khi áp dụng vào project có nhiều document hơn.

#### 3. Timeline triển khai
- **Tuần 1:** Implement hierarchical chunking + metadata extraction (version/ngày hiệu lực) cho corpus thật của project.
- **Tuần 2:** Implement hybrid search (BM25 + Dense + RRF) + version-aware filtering trước rerank; viết test set Q&A riêng (tối thiểu 15-20 câu, có câu multi-hop và câu "version conflict" như trong lab).
- **Tuần 3:** Chạy RAGAS evaluation, đo latency breakdown, tối ưu enrichment bằng async batching nếu cần giảm thời gian index.
