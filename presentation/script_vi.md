# Kịch bản thuyết trình (tiếng Việt) – ~10 phút

**Đề tài:** Merging BERT Experts with Weight-Ensembling MoE
**Nhóm 1:** Nguyễn Bá Anh (20261218M) – Nguyễn Thị Nhã Linh (20261262M)
**Giảng viên:** Nguyễn Kiêm Hiếu

Kịch bản này đi theo `presentation_revised.pdf`. Slide 1–15 là nội dung chính (≈ 10 phút, trung bình 40 giây mỗi slide). Slide 16–23 (backup) và 24 (tài liệu tham khảo) chỉ dùng khi được hỏi.

---

## Slide 1 – Trang bìa (≈ 0:20)

Xin chào thầy và các bạn. Chúng em là nhóm 1, gồm Nhã Linh và Bá Anh, dưới sự hướng dẫn của thầy Nguyễn Kiêm Hiếu. Hôm nay nhóm trình bày về việc hợp nhất các chuyên gia BERT bằng Weight-Ensembling Mixture of Experts, gọi tắt là WEMoE, với router được huấn luyện bằng một lượng nhỏ dữ liệu có nhãn, trên năm tác vụ GLUE.

---

## Slide 2 – Motivation (≈ 0:35)

Năm chuyên gia fine-tune riêng đạt điểm **78,2**, nhưng cần **547,4 triệu tham số**. Task Arithmetic hợp nhất tĩnh chỉ cần 109,5 triệu tham số, nhưng điểm giảm còn **57,4**.

Câu hỏi của nhóm: *trọng số MLP phụ thuộc đầu vào có lấy lại được chất lượng với một mô hình hợp nhất, và tốn ít bộ nhớ hơn năm chuyên gia hay không?*

Đây là bản chuyển thể sang BERT và GLUE, không phải tái hiện thí nghiệm CLIP trên ảnh của bài báo.

---

## Slide 3 – Theory (≈ 0:45)

Hình bên trái lấy từ bài báo: các khối ngoài MLP được hợp nhất tĩnh, còn MLP được nâng cấp thành mô-đun WEMoE ở hình bên phải.

Task vector τ_e bằng W_e trừ W₀, là phần kiến thức riêng của tác vụ, và chính là "chuyên gia" trong mô hình. Các khối ngoài MLP dùng công thức tĩnh θ₀ cộng λ nhân tổng τ_e. Khối MLP dùng W(x) bằng W₀ cộng tổng w_e(x) nhân τ_e, nên trọng số thay đổi theo từng đầu vào.

Các hệ số w_e **không được chuẩn hóa**, nên router có thể khuếch đại hoặc trừ một task vector.

---

## Slide 4 – Method (≈ 0:55)

Đây là kiến trúc nhóm triển khai trên BERT. Dòng giữa là luồng dữ liệu: token đi qua embedding, mười hai tầng encoder, pooler, rồi đến head của tác vụ t.

Trong mỗi tầng, attention và các thành phần ngoài MLP dùng Task Arithmetic với λ bằng 0,3. Hai lớp fc1 và fc2 của MLP lấy trọng số từ từ điển gồm W₀ và năm task vector, được trộn theo w^ℓ(x) do router của tầng đó sinh ra.

Nhóm thử hai phạm vi: *full* ensemble cả fc1 và fc2; *intermediate-only* chỉ ensemble fc1.

Điểm cần nhớ: **các chuyên gia không chạy khi suy luận**. Hệ thống chỉ chạy một lượt forward trên một mô hình, và task id chọn head.

---

## Slide 5 – Components (≈ 0:45)

Hàng trên là chuyên gia: BERT được fine-tune toàn bộ cho một tác vụ, sau đó đóng băng. WEMoE chỉ lấy lại task vector và head của nó.

Hàng dưới là router: hai lớp tuyến tính với ReLU ở giữa, chạy trên **từng token**, rồi lấy trung bình trọng số trên các token không phải padding. Mười hai router có tổng cộng 7,1 triệu tham số.

Đầu ra **không có softmax và không có top-k**: đây là vector số thực, không phải phân phối xác suất. Bias đầu ra khởi tạo bằng λ, nên mô hình bắt đầu gần với Task Arithmetic.

---

## Slide 6 – Pipeline (≈ 0:40)

Quy trình có bốn giai đoạn. Một, fine-tune từng chuyên gia bằng cross-entropy. Hai, hợp nhất: tính task vector và Task Arithmetic, không huấn luyện gì. Ba, FSFT: tối thiểu cross-entropy trên k mẫu mỗi tác vụ, và **chỉ cập nhật router**. Bốn, suy luận: tạo W(x) theo đầu vào rồi lấy lớp có xác suất cao nhất.

Bài báo huấn luyện router không nhãn bằng entropy trên tập test, còn nhóm dùng cross-entropy có nhãn.

---

## Slide 7 – Data boundary (≈ 0:30)

Để tránh rò rỉ, mỗi tập train GLUE chia làm ba phần: `expert_train` để fine-tune chuyên gia; `fewshot_pool` 256 mẫu mỗi tác vụ chỉ để huấn luyện router; và `dev`, tối đa 1.000 mẫu, để chọn epoch, λ và bước dừng. Validation chính thức chỉ dùng để tính kết quả cuối cùng.

---

## Slide 8 – Experiments (≈ 0:25)

Nhóm so sánh từ thấp đến cao: lớp đa số, Simple Averaging, Task Arithmetic với λ chọn trên dev, WEMoE trước và sau FSFT, và năm chuyên gia làm mức tham chiếu.

Thang điểm: MRPC là trung bình accuracy và F1; CoLA là MCC; còn lại là accuracy; tất cả nhân 100.

---

## Slide 9 – Main result (≈ 0:50)

Lớp đa số đạt 45,8; Simple Averaging 53,3; Task Arithmetic 57,4. Khởi tạo WEMoE chỉ đạt 57,8, nên cải thiện không đến từ khởi tạo. Sau FSFT, bản full MLP với k bằng 256 đạt **69,9 ± 0,1**, còn năm chuyên gia đạt 78,2.

Nếu đo bằng "gain over majority", tức phần khoảng cách từ lớp đa số đến chuyên gia lấy lại được, Task Arithmetic đạt **18%**, còn WEMoE cộng FSFT đạt **59%**.

Các giá trị là trung bình và độ lệch chuẩn trên ba seed router.

---

## Slide 10 – Few-shot budget (≈ 0:35)

Với k bằng 16, điểm là 60,8; k bằng 64 là 64,6; k bằng 256 là 69,9 và độ lệch chuẩn chỉ 0,1.

Với k nhỏ, router overfit sớm và kết quả dao động mạnh giữa các seed: độ lệch chuẩn CoLA MCC là 14,1 và 17,8. Từ k bằng 16 lên 64, điểm chỉ tăng 3,8 nhưng gain gần như gấp đôi, vì một phần điểm ở k bằng 16 đến từ MRPC và RTE nằm ở mức đoán hằng.

---

## Slide 11 – Per-task results (≈ 0:40)

Bản full MLP vượt lớp đa số ở MRPC, CoLA, SST-2 và QNLI, nhưng **RTE chỉ 52,0, thấp hơn majority 52,7**.

Bản intermediate-only thì ngược lại: RTE lên 59,0, nhưng MRPC chỉ 66,5 / 76,6, **thấp hơn majority** 68,4 / 81,2. Cả hai bản vẫn dưới năm chuyên gia ở mọi tác vụ.

Tập validation của RTE có 277 mẫu và MRPC có 408 mẫu, sai số chuẩn khoảng 3 và 2 điểm, nên một phần chênh lệch có thể là nhiễu.

---

## Slide 12 – Storage trade-off (≈ 0:45)

Năm chuyên gia: 547,4 triệu tham số. WEMoE full: 400,0 triệu, tức 73,1%. Intermediate-only: 258,4 triệu, tức 47,2%, với điểm 69,1, gần bằng 69,9 của bản full, và ít hơn khoảng 35% tham số.

Lưu ý quan trọng: mỗi tác vụ giữ head riêng nên ta biết task id lúc suy luận. Khi đó năm chuyên gia vừa chính xác hơn, vừa chỉ lớn hơn bản full khoảng 37%. Vì vậy, lý do dùng WEMoE dựa vào bản intermediate-only, hoặc vào bối cảnh không biết tác vụ.

Thời gian FSFT trên một T4 là 14–16 phút và 8,5 phút; đó là thời gian huấn luyện, không phải độ trễ suy luận.

---

## Slide 13 – Routing và nguyên nhân RTE (≈ 0:50)

Biểu đồ là trọng số routing trung bình. Mẫu hình khá giống nhau giữa các tác vụ, tương quan 0,82, nhưng loại tác vụ đầu vào vẫn giải thích khoảng 58% phương sai. Trọng số trải từ −3,0 đến 4,4, tức router thực sự trừ và khuếch đại task vector.

Về RTE: router full MLP gán task vector RTE gần 0 ngay trên đầu vào RTE, chỉ 0,01, so với 1,37 của MRPC, và còn trừ nó ở tầng 11–12, khoảng −1,2. Bản intermediate-only gán 0,96 và đạt 59,0.

Hình chỉ từ một cấu hình, seed 0. Ablation routing độc lập đầu vào chưa có kết quả, nên chưa kết luận được việc phụ thuộc đầu vào là nguyên nhân của mức tăng điểm.

---

## Slide 14 – Limitations (≈ 0:30)

Một, RTE vẫn dưới majority ở bản full, và chuyên gia RTE cũng yếu, 63,2. Hai, k nhỏ không ổn định. Ba, chọn mô hình dùng tới 1.000 mẫu dev, nhiều hơn 256 mẫu huấn luyện. Bốn, chuyên gia và cách chia dữ liệu chỉ có một seed.

Khác biệt về dữ liệu và cách huấn luyện router khiến việc so sánh với bài báo chỉ mang tính định tính.

---

## Slide 15 – Conclusion (≈ 0:30)

Trong lần chạy đầy đủ đầu tiên, huấn luyện router với 256 mẫu mỗi tác vụ nâng điểm từ 57,4 lên 69,9, so với 78,2 của năm chuyên gia. Full MLP và intermediate-only có điểm gần nhau nhưng khác về lưu trữ và từng tác vụ. Hướng tiếp theo: thêm seed, few-shot chặt chẽ hơn, ablation routing và độ trễ suy luận.

Xin cảm ơn thầy và các bạn, nhóm em sẵn sàng nhận câu hỏi.

---

## Dự phòng khi được hỏi (slide 16–23)

- **Slide 16 – Transformer:** encoder gồm self-attention và FFN; decoder thêm masked self-attention và cross-attention. Mỗi lớp con có residual và LayerNorm. Công thức sin/cos là của Transformer gốc.
- **Slide 17 – BERT:** chỉ dùng encoder, ngữ cảnh hai chiều. Embedding bằng tổng token, segment và position; position của BERT là **học được**, không phải sin/cos. MLM và NSP chỉ thuộc giai đoạn pre-training; nhóm không dùng hai head này.
- **Slide 18 – Fine-tuning:** cả năm tác vụ dùng chung một công thức huấn luyện và đều là phân loại nhị phân, nên **không có regression head**. Head là Dropout cộng Linear trên `pooler_output`.
- **Slide 19 – Weight vs. logit:** logit ensembling trộn đầu ra và cần một lượt forward cho mỗi mô hình; WEMoE trộn trọng số trong MLP và chỉ cần một lượt. Với một lớp tuyến tính, hai cách tương đương về mặt đại số; GELU và 12 tầng xếp chồng làm toàn mô hình khác. **Nhóm không chạy logit ensembling**, nên đây là so sánh khái niệm, không có số đo.
- **Slide 20 – Router và MoE:** MoE cổ điển trộn đầu ra và có thể dùng top-k. Ở đây routing là **dense**: cả năm task vector tham gia ở mọi tầng, trọng số có thể âm, không phải tổ hợp lồi.
- **Slide 21 – Cách tính điểm:** điểm trung bình không trọng số của năm tác vụ, thang ×100; độ lệch chuẩn trên seed router 0, 1, 2, còn chuyên gia và cách chia dữ liệu cố định.
- **Slide 22 – Huấn luyện router:** AdamW, lr 10⁻⁴, batch 16 mỗi tác vụ, tối đa 500 bước; chọn checkpoint trên dev mỗi 50 bước, tối đa 1.000 mẫu mỗi tác vụ.
- **Slide 23 – Bài báo và đề tài này:** bài báo dùng CLIP-ViT trên 8 tác vụ ảnh và huấn luyện router bằng entropy trên dữ liệu test không nhãn. Nhóm dùng BERT-base trên 5 tác vụ GLUE và cross-entropy trên k mẫu có nhãn. Bài báo không có kết quả trên GLUE nên so sánh chỉ mang tính định tính.
- **Phạm vi:** đây là kết quả của lần chạy đầu tiên, không phải bảo đảm tổng quát hay tái hiện hoàn chỉnh WEMoE.
