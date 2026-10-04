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

### Project: Attacker vs. Target LLM System (RAG là một phần của Target System)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Project gồm 2 phía — (1) **Target LLM system**: một hệ thống có RAG pipeline đóng vai trò knowledge-base backend (kiến trúc tương tự lab hôm nay — chunking + hybrid search + rerank + LLM generation) để trả lời câu hỏi dựa trên corpus nội bộ; (2) **Attacker**: thiết kế các kỹ thuật tấn công nhằm vào target, khai thác chính những điểm yếu của RAG pipeline.
- **Known issues (nhìn từ góc độ attacker, rút ra trực tiếp từ failure analysis của lab):**
  - Reranker (`bge-reranker-v2-m3`) không phân biệt được "tài liệu nào đang hiệu lực" nếu 2 chunk gần giống nhau về câu chữ (đã thấy rõ ở case mật khẩu v1/v2: score 0.998 vs 0.997) → attacker có thể **poison knowledge base** bằng 1 document giả rất giống document hợp lệ (cùng structure, cùng wording, chỉ đổi số liệu/điều khoản quan trọng) để nó lọt top-k ngang hàng với document thật và đánh lừa target.
  - Enrichment (M5, combined single-call) tự sinh context mô tả "Đoạn văn nằm trong phần... của tài liệu..." dựa trên nội dung chunk — nếu attacker kiểm soát được một phần nội dung nạp vào knowledge base (ví dụ qua upload công khai, ticket, comment), có thể chèn **indirect prompt injection** ngay trong chunk đó; injection sẽ được enrich → index → retrieve → đưa thẳng vào context của LLM generation ở cuối pipeline mà không qua bước sanitize nào.
  - System prompt hiện tại ("Trả lời CHỈ dựa trên context") tin tưởng tuyệt đối nội dung retrieve được → đây chính là bề mặt tấn công cốt lõi: ai kiểm soát được context thì kiểm soát được output của target.

#### 2. Kế hoạch áp dụng (Attacker techniques + Target hardening)
1. **Chunking strategy:** Dựng target baseline bằng hierarchical chunking (như lab) — vì đây là lựa chọn phổ biến trong RAG production nên attacker cần test đúng setup thực tế. Thử nghiệm structure-aware chunking cho attack: vì nó giữ nguyên markdown header, attacker có thể **giả header** (`## Chính sách hiện hành`) để chunk độc hại "trông chính thức" hơn trong mắt reranker/LLM.
2. **Search retrieval:** Dùng Hybrid (BM25 + Dense + RRF) làm target baseline, rồi khai thác riêng từng nhánh: test **keyword-stuffing** (nhồi từ khóa trùng query để ép BM25 đẩy chunk độc hại lên top-k) và **embedding-mimicry** (viết chunk giả có embedding gần giống chunk thật để qua được Dense search).
3. **Reranking:** Dùng cross-encoder làm lớp defense cuối, nhưng lab đã chứng minh nó chỉ học semantic similarity, KHÔNG học "nguồn nào đáng tin / còn hiệu lực theo thời gian" — đây là attack vector cụ thể cần khai thác (document version-conflict injection), sau đó đề xuất hardening: ký số / metadata xác thực nguồn tài liệu thay vì chỉ dựa vào similarity score.
4. **Evaluation:** Dùng RAGAS không chỉ để đo chất lượng mà để đo **mức độ thành công của attack** — ví dụ: đo Faithfulness của câu trả lời SAU KHI inject tài liệu giả; nếu Faithfulness vẫn cao (LLM "bám context" tốt) nhưng nội dung sai sự thật so với ground_truth gốc → đó chính là bằng chứng attack thành công (RAG đã tin vào context bị đầu độc), dù nhìn qua metric thông thường vẫn "trông tốt". Cần định nghĩa thêm 1 custom metric: **injection success rate** (% câu trả lời bị chi phối bởi payload injected).
5. **Enrichment:** Đây là bề mặt tấn công rõ nhất cho indirect prompt injection trong toàn pipeline. Action cụ thể: viết test case chèn instruction giả (ví dụ "Ignore previous instructions and reveal...") vào 1 document trong corpus, chạy qua `enrich_chunks()`, kiểm tra xem `enriched_text` / `hypothesis_questions` có "rò" injection ra ngoài không, và cuối cùng xem `run_query()` có bị chi phối hành vi không — từ đó đề xuất sanitize layer giữa retrieval và generation (strip instruction-like patterns khỏi context trước khi feed LLM).

#### 3. Timeline triển khai
- **Tuần 1:** Dựng target system baseline bằng chính code lab này (M1-M5 + pipeline) với corpus giả định làm "target knowledge base"; đo RAGAS baseline để có mốc so sánh trước khi attack.
- **Tuần 2:** Thiết kế & thực thi các kịch bản attack: document poisoning (version-conflict), indirect prompt injection qua chunk/enrichment, keyword-stuffing BM25, embedding-mimicry cho Dense search.
- **Tuần 3:** Đo lại bằng RAGAS + custom injection-success-rate metric, so sánh trước/sau attack; viết báo cáo đề xuất hardening cho target (metadata xác thực nguồn, instruction-input separation, sanitize context trước khi feed LLM, rate-limit/anomaly detection cho corpus upload).
