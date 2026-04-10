# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phan Anh Khôi
**Nhóm:** [Tên nhóm]
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Hai vector đại diện cho 2 từ/câu/ảnh... có high cosine similarity có nghĩa là góc giữa 2 vector nhỏ, 2 vector có hướng gần giống nhau, nội dung/ngữ nghĩa của hai từ/câu/ảnh rất tương đồng dù có thể dùng từ khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: Con chó nghiệp vụ này rất cừ.
- Sentence B: Con cảnh khuyển này rất giỏi.
- Tại sao tương đồng: 2 câu về mặt ngữ nghĩa giống nhau, dùng các từ đồng nghĩa.

**Ví dụ LOW similarity:**
- Sentence A: Con chó nghiệp vụ này rất cừ.
- Sentence B: Mèo lười thèm cá rán.
- Tại sao khác: Hai câu không có gì giống nhau về mặt ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Vì cosine similarity chỉ quan tâm đến hướng (ngữ nghĩa) của vector, không bị ảnh hưởng bởi độ dài vector như Euclidean distance, nên phù hợp hơn để so sánh ý nghĩa câu.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* (tổng số ký tự - số ký tự chunk cuối) / (chunk_size - overlap) + 1 = (10000 - 500) / (500 - 50) + 1 = 22
> *Đáp án:* 22 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Số chunks = ⌈9500 / 400⌉ + 1 = 24 + 1 = 25 chunks — nhiều hơn 3 chunks so với overlap=50. Overlap lớn hơn giúp ngữ cảnh ở ranh giới giữa các chunk được giữ lại nhiều hơn, thông tin quan trọng nếu ở ranh giới một chunk có thể có đấy đủ context ở chunk tiếp theo, giúp tránh mất thông tin ở điểm nối.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...] 

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Vì tài chính rất khó và mang tính thời sự.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| NVIDIA | FixedSizeChunker (`fixed_size`) | | | |
| TESLA | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy] 
SentenceChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?* 
SentenceChunker chia văn bản thành các câu dựa trên dấu câu như “.”, “;”, “:”, hoặc xuống dòng. Sau đó, nó có thể gộp nhiều câu lại thành một chunk sao cho không vượt quá giới hạn độ dài (token/character). Mỗi chunk giữ nguyên cấu trúc ngữ nghĩa tự nhiên của câu thay vì cắt giữa chừng. Nhờ vậy, thông tin trong từng chunk vẫn đầy đủ và dễ hiểu đối với model.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*
Báo cáo tài chính thường được viết theo các câu hoặc đoạn có cấu trúc rõ ràng, mỗi câu mang một ý nghĩa hoàn chỉnh (ví dụ: mô tả chỉ số, giải thích biến động). SentenceChunker giúp giữ nguyên các đơn vị ý nghĩa này, tránh việc cắt ngang số liệu hoặc giải thích quan trọng. Điều này giúp embedding phản ánh đúng ngữ cảnh tài chính và cải thiện chất lượng truy vấn.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** __ / __

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
