# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Bùi Gia Chính (MSSV: 2A202602693)
**Khóa:** K4 - Track 3A

---

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7500 | 0.6833 | -0.0667 |
| Answer Relevancy | 0.7062 | 0.6909 | -0.0153 |
| Context Precision | 0.9250 | 0.9708 | +0.0458 |
| Context Recall | 0.9083 | 0.8833 | -0.0250 |

**Quan sát tổng quát:** Production chỉ thắng Naive ở Context Precision (reranker lọc nhiễu tốt hơn dense-only). Ba metric còn lại thấp hơn — ngược lại kỳ vọng ban đầu rằng hybrid+rerank+enrichment luôn cải thiện mọi mặt. Điều tra bottom-5 bên dưới cho thấy nguyên nhân chính không phải chunking/search kém, mà là **ambiguity giữa các phiên bản chính sách** và **multi-hop retrieval bị giới hạn bởi RERANK_TOP_K=3**.

## Bottom-5 Failures

### #1
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness (0.0)
- **Error Tree:**
  - Output sai? → Có, trả lời "Không tìm thấy" dù dữ liệu tồn tại.
  - Context đúng? → **Có** — đã verify bằng cách trace lại pipeline thủ công: top-3 sau rerank là `mat_khau_v2.md` (12 ký tự, score 0.998), `mat_khau_v1.md` (8 ký tự, score 0.997), và 1 chunk enrichment-context khác của v2. Cả 2 phiên bản đều lọt top-3 với score gần như bằng nhau.
  - Query OK? → Có, query rõ ràng.
  - **Root cause:** Chunk của v1 và v2 có cấu trúc câu giống nhau đến ~99%, chỉ khác số (8 vs 12 ký tự) → cross-encoder không có tín hiệu nào để phân biệt "chính sách nào đang hiệu lực" (không có date/version trong embedding signal). LLM nhận context chứa 2 câu trả lời MÂU THUẪN (8 vs 12 ký tự) nên theo đúng system prompt ("Nếu không có → nói 'Không tìm thấy'") đã từ chối trả lời để tránh đưa thông tin sai — **safe refusal**, nhưng RAGAS chấm faithfulness=0.0 vì câu "Không tìm thấy" không entail được từ ground truth.
- **Suggested fix:** Thêm metadata `is_current: bool` / `superseded_by` khi chunking hoặc enrichment (M5 `extract_metadata`), rồi filter/boost chunk "current" trước khi đưa vào context; hoặc rewrite system prompt để yêu cầu LLM ưu tiên tài liệu có version mới nhất khi gặp mâu thuẫn, thay vì từ chối.

### #2
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Giống #1 — cùng root cause (v1 "90 ngày" vs v2 "120 ngày" cùng lọt top-3, LLM từ chối vì mâu thuẫn).
- **Suggested fix:** Giống #1.

### #3
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Không tìm thấy."
- **Worst metric:** faithfulness (0.0)
- **Error Tree:**
  - Context đúng? → Một phần. Dense search (top-5) tìm được cả `thu_viec.md` (quy tắc 85% thử việc) và `bang_luong_2024.md` (khung lương Junior) riêng lẻ, nhưng đây là câu hỏi **multi-hop** — cần GHÉP 2 fact từ 2 document khác nhau trong cùng context window.
  - **Root cause:** `RERANK_TOP_K=3` (config.py) quá nhỏ cho multi-hop — khi 2 chunk cần thiết nằm ở 2 nguồn khác nhau, rerank top-3 không đủ chỗ giữ cả 2 cùng lúc nếu có chunk khác điểm cao hơn chen vào.
- **Suggested fix:** Tăng `RERANK_TOP_K` lên 5 cho câu hỏi multi-hop, hoặc thêm bước query decomposition (tách câu hỏi multi-hop thành 2 sub-query) trước khi retrieve.

### #4
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** "Nhân viên Senior có 9 năm thâm niên sẽ được 18 ngày phép năm... Không có thông tin cụ thể về lương trong ngữ cảnh đã cho."
- **Worst metric:** answer_relevancy (0.0)
- **Error Tree:** Output sai một phần (đúng số ngày phép, thiếu lương) → Context đúng một phần (chunk nghỉ phép có trong top-3, chunk lương Senior bị rerank đẩy ra ngoài top-3) → **Root cause:** multi-hop giống #3, cùng giới hạn `RERANK_TOP_K=3`.
- **Suggested fix:** Giống #3.

### #5
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Cần Tổng Giám đốc (CEO) phê duyệt." (đúng nội dung, chỉ ngắn hơn ground truth)
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Output **đúng về nội dung** (CEO phê duyệt là chính xác theo data) → Context đúng → Query OK → **Root cause:** đây nhiều khả năng là **nhiễu từ RAGAS judge** (judge LLM — gpt-4o-mini qua proxy ShopAIKey — chấm faithfulness nghiêm khắc/không ổn định với câu trả lời ngắn, thiếu số liệu "50.000.000 VNĐ" tường minh dù ý đúng), không phải lỗi thật của pipeline.
- **Suggested fix:** Yêu cầu LLM answer-generation luôn trích dẫn số liệu cụ thể từ context (ví dụ thêm instruction "luôn nêu số liệu/ngưỡng cụ thể nếu context có") để tăng độ "bám context" quan sát được bởi judge; đồng thời nên chạy RAGAS evaluate 2-3 lần lấy trung bình để giảm variance của judge.

## Latency Breakdown (Production Pipeline)

| Step | Thời gian | Ghi chú |
|------|-----------|---------|
| Chunking (M1, 107 chunks / 26 docs) | 0.1s | Hierarchical chunking, CPU-only |
| Enrichment (M5, 1 API call/chunk) | 574.3s | 107 chunks × ~5.4s/call (combined mode, qua proxy ShopAIKey) — chiếm **52%** tổng thời gian, là bottleneck lớn nhất |
| Indexing (M2, BM25 + Dense bge-m3) | 52.5s | Chủ yếu là encode embedding 107 chunks bằng bge-m3 (1024-dim) |
| Reranker load (M3) | 0.0s | Model đã pre-download, load từ cache |
| 20 queries (search + rerank + LLM answer) | ~56s | Bao gồm cả gọi LLM sinh câu trả lời |
| RAGAS evaluation (4 metrics × 20 câu) | 65.5s | 80 lần gọi judge LLM |
| **Tổng** | **1109.0s (~18.5 phút)** | |

**Nhận xét:** Enrichment (M5) là bottleneck rõ ràng nhất vì chạy tuần tự 1 call/chunk. Có thể cải thiện bằng cách batch/parallelize các API call (ví dụ `asyncio` + `AsyncOpenAI`) để giảm latency tổng thể mà không đổi logic.

## Case Study (presentation)

**Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?

**Error Tree walkthrough:**
1. Output đúng? → Sai — trả lời "Không tìm thấy" dù data có sẵn.
2. Context đúng? → Đúng — trace thủ công cho thấy chunk `mat_khau_v2.md` (12 ký tự, đúng) nằm ở rank #1 sau rerank (score 0.998).
3. Query rewrite OK? → Có, không cần rewrite.
4. Fix ở bước: **Metadata/version filtering trước bước đưa context vào LLM** (giữa M2/M3 và bước generation) — đây là lỗ hổng duy nhất trong pipeline, mọi bước upstream (chunking, search, rerank) đều hoạt động đúng như kỳ vọng.

**Nếu có thêm 1 giờ:**
- Thêm field `policy_version`/`is_superseded` vào metadata khi enrich (M5 `extract_metadata`), filter chunk cũ ra trước khi build context cho câu hỏi "chính sách hiện hành".
- Tăng `RERANK_TOP_K` thành biến theo loại câu hỏi (phát hiện multi-hop qua từ khóa "và", "bao nhiêu...và..." → tăng top_k lên 5-6).
- Chạy RAGAS 3 lần lấy trung bình để tách biệt lỗi pipeline thật với nhiễu của judge LLM.
