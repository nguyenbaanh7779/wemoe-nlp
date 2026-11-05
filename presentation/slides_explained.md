# Giải thích từng slide của bộ WEMoE

Tài liệu này dành cho người mới đọc repo. Bạn nên đọc nó song song với `presentation_revised.pdf` (24 slide). Với mỗi slide, tài liệu cho biết slide muốn nói gì, giải thích các khái niệm và chỉ ra nơi kiểm tra trong repo.

Đây là tài liệu tra cứu nội bộ, không phải phần trình chiếu.

## Đọc phần này trước

### Bài toán trong một đoạn

Ta có năm mô hình BERT, mỗi mô hình được huấn luyện riêng cho một tác vụ NLP. Giữ cả năm thì tốn bộ nhớ. Cộng gộp trọng số thành một mô hình thì nhỏ gọn nhưng mất nhiều chất lượng. WEMoE là cách ở giữa: giữ một mô hình chung, nhưng trọng số của các lớp MLP được tính lại **cho từng câu đầu vào**, nhờ một mạng nhỏ gọi là router. Đề tài này kiểm tra xem cách đó lấy lại được bao nhiêu chất lượng và tiết kiệm được bao nhiêu bộ nhớ.

### Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **Expert (chuyên gia)** | Một BERT đã fine-tune cho một tác vụ |
| **Fine-tune** | Huấn luyện tiếp một mô hình đã pre-train trên dữ liệu của tác vụ cụ thể |
| **Task vector `τ_e`** | Phần trọng số thay đổi sau fine-tune: `θ_e − θ_0` |
| **Task Arithmetic (TA)** | Hợp nhất tĩnh: `θ_0 + λ Σ τ_e`, cùng một trọng số cho mọi đầu vào |
| **Router** | Mạng nhỏ nhìn vào đầu vào và sinh hệ số `w_e(x)` cho từng task vector |
| **MLP của BERT** | Hai lớp tuyến tính trong mỗi tầng encoder; trong code là `intermediate.dense` (fc1) và `output.dense` (fc2) |
| **FSFT** | Few-Shot Fine-Tuning: huấn luyện **chỉ router** bằng `k` mẫu có nhãn mỗi tác vụ |
| **`k`** | Số mẫu có nhãn mỗi tác vụ dùng cho FSFT |
| **`λ`** | Hệ số của Task Arithmetic; lần chạy này chọn được 0,3 trên tập dev |
| **Majority** | Baseline luôn đoán lớp phổ biến nhất |
| **Seed** | Hạt ngẫu nhiên; lặp lại với nhiều seed để xem kết quả có ổn định không |

### Năm tác vụ GLUE

Cả năm đều là phân loại hai lớp.

| Tác vụ | Đầu vào | Dự đoán | Điểm |
|---|---|---|---|
| MRPC | Cặp câu | Có tương đương không | Trung bình accuracy và F1 |
| RTE | Cặp câu | Có suy ra được không | Accuracy |
| CoLA | Một câu | Có đúng ngữ pháp không | MCC |
| SST-2 | Một câu | Tích cực hay tiêu cực | Accuracy |
| QNLI | Câu hỏi và một câu | Câu có trả lời được câu hỏi không | Accuracy |

### Bản đồ repo

| Muốn xem | Mở file |
|---|---|
| Kiến trúc WEMoE, router | [src/wemoe_nlp/wemoe.py](../src/wemoe_nlp/wemoe.py) |
| Task vector, Task Arithmetic | [src/wemoe_nlp/merging.py](../src/wemoe_nlp/merging.py) |
| Huấn luyện chuyên gia | [src/wemoe_nlp/experts.py](../src/wemoe_nlp/experts.py) |
| Huấn luyện router (FSFT) | [src/wemoe_nlp/fsft.py](../src/wemoe_nlp/fsft.py) |
| Chia dữ liệu | [src/wemoe_nlp/data.py](../src/wemoe_nlp/data.py) |
| Cách tính điểm | [src/wemoe_nlp/metrics.py](../src/wemoe_nlp/metrics.py) |
| Siêu tham số | [configs/default.yaml](../configs/default.yaml) |
| Bảng kết quả | [results/summary.md](../results/summary.md) |
| Phân tích kết quả | [docs/report/README.md](../docs/report/README.md) |

### Mức độ kiểm chứng của số liệu

Không phải số nào cũng kiểm tra được giống nhau:

- **Kiểm tra được từ code hoặc file kết quả:** số tham số, cấu hình huấn luyện, baseline, 7 run trong `results/wemoe/`.
- **Chỉ có trong báo cáo hoặc bảng tổng hợp:** các số routing ở slide 13 và trung bình 3 seed của cấu hình full MLP, `k=256`. Repo chỉ có seed 0 của cấu hình này (điểm 69,7); seed 1 và 2 nằm trên Google Drive.

---

# Phần chính (slide 1–15)

## Slide 1: Trang bìa

Giới thiệu đề tài, nhóm và giảng viên. Tiêu đề phụ "Few-shot router training across five GLUE tasks" cho biết hai điểm riêng của đề tài: router được huấn luyện bằng ít dữ liệu có nhãn, và thử trên năm tác vụ GLUE.

## Slide 2: Motivation

**Ý chính:** cần một mô hình vừa nhỏ vừa giữ được chất lượng.

Ba ô trên slide là ba lựa chọn:

| Lựa chọn | Điểm | Tham số | Nhận xét |
|---|---|---|---|
| Năm chuyên gia | 78,2 | 547,4M | Tốt nhất nhưng phải lưu năm checkpoint |
| Task Arithmetic | 57,4 | 109,5M | Gọn nhưng mất nhiều chất lượng |
| WEMoE | `?` | `?` | Câu hỏi nghiên cứu của đề tài |

Khung "Scope" nhắc rằng bài báo gốc của WEMoE làm trên ảnh với CLIP. Đề tài này chuyển phương pháp sang văn bản với BERT, nên không phải là tái hiện thí nghiệm của bài báo.

**Kiểm tra:** [results/summary.md](../results/summary.md), bảng kết quả và bảng tham số.

## Slide 3: Theory

**Ý chính:** tách phần chung và phần riêng của tác vụ, rồi trộn phần riêng theo từng đầu vào.

Hình lấy từ bài báo gốc:

- **Hình trái:** các khối ổn định (Attention, LayerNorm) được hợp nhất tĩnh, còn MLP được nâng cấp (upcycling) thành mô-đun WEMoE.
- **Hình phải:** bên trong mô-đun WEMoE. Router sinh hệ số, nhân với từng task vector, rồi cộng vào trọng số chung `θ_0`.
- Biểu tượng bông tuyết nghĩa là đóng băng, ngọn lửa nghĩa là được huấn luyện.

Ba công thức:

```text
τ_e  = W_e − W_0                    task vector của chuyên gia e
θ    = θ_0 + λ Σ_e τ_e              các khối ngoài MLP, cố định
W(x) = W_0 + Σ_e w_e(x) · τ_e       MLP, thay đổi theo đầu vào x
```

Task Arithmetic dùng cùng một `λ` cho mọi câu. WEMoE thay `λ` bằng `w_e(x)` riêng cho từng câu và từng tầng. Vì vậy, "chuyên gia" trong WEMoE thực chất là một **task vector được lưu sẵn**, không phải một mô hình đang chạy.

Hệ số **không chuẩn hóa** nghĩa là không có softmax, nên hệ số có thể âm (trừ task vector) hoặc lớn hơn 1 (khuếch đại).

Lưu ý: hình của bài báo vẽ ViT, nơi LayerNorm đứng trước Attention. BERT đặt Add&Norm sau mỗi khối con, nên kiến trúc thật được vẽ lại ở slide 4.

**Kiểm tra:** [merging.py:18-23](../src/wemoe_nlp/merging.py#L18-L23), [wemoe.py:8-15](../src/wemoe_nlp/wemoe.py#L8-L15).

## Slide 4: Method

**Ý chính:** đây là kiến trúc được cài đặt thật trên BERT.

Đọc sơ đồ theo màu:

- **Xanh:** đóng băng, không huấn luyện.
- **Vàng:** trọng số thay đổi theo đầu vào.
- **Đỏ:** được huấn luyện trong FSFT.

Luồng dữ liệu đi từ trái sang phải:

1. Câu được token hóa. Hệ thống cũng cần biết tác vụ `t` của câu.
2. Token đi qua embedding, có trọng số Task Arithmetic.
3. Mỗi trong 12 tầng encoder thực hiện:
   - Attention và Add&Norm, với trọng số Task Arithmetic.
   - Router đọc trạng thái token và sinh `w^ℓ(x)` có 5 giá trị.
   - fc1 và fc2 dùng trọng số `W_0 + Σ w_e τ_e`, cùng chung một `w^ℓ(x)`.
4. Pooler lấy biểu diễn `[CLS]`, rồi head của tác vụ `t` cho ra logits.

Có hai phạm vi được thử:

| Phạm vi | fc1 | fc2 |
|---|---|---|
| Full | WEMoE | WEMoE |
| Intermediate-only | WEMoE | Task Arithmetic |

Hiểu lầm thường gặp: **năm chuyên gia không chạy song song khi suy luận.** Chỉ có một lượt forward trên một mô hình. Head được chọn bằng task id, không phải do router.

**Kiểm tra:** `build_wemoe` tại [wemoe.py:278-324](../src/wemoe_nlp/wemoe.py#L278-L324), `forward` tại [wemoe.py:241-247](../src/wemoe_nlp/wemoe.py#L241-L247).

## Slide 5: Components

**Ý chính:** chuyên gia được huấn luyện rồi đóng băng; chỉ router tiếp tục học.

**Chuyên gia** gồm BERT, pooler và head (Dropout + Linear). Sau khi fine-tune, WEMoE chỉ lấy task vector và head từ nó.

**Router** là một mạng rất nhỏ:

```text
r(h) = A_2 · ReLU(A_1·h + b_1) + b_2
w(x) = Σ_i m_i · r(h_i) / Σ_i m_i
```

- `h_i` là trạng thái của token thứ `i`, `d` là hidden size của BERT.
- `m_i` bằng 1 nếu là token thật, bằng 0 nếu là padding.
- Router chạy **trên từng token rồi mới lấy trung bình**, không phải lấy trung bình token trước. Vì có ReLU nên hai cách này cho kết quả khác nhau.

Ba điểm cần nhớ:

1. **Không softmax, không top-k:** đầu ra là vector số thực, không phải xác suất.
2. **Khởi tạo:** `b_2 = λ` và trọng số rất nhỏ, nên ban đầu `w_e(x) ≈ λ`. Mô hình xuất phát gần Task Arithmetic.
3. **Chung một `w`:** trong một tầng, fc1 và fc2 dùng cùng một vector hệ số, cho cả weight lẫn bias.

Tổng 12 router có 7,1M tham số, rất nhỏ so với toàn mô hình.

**Kiểm tra:** `Router` tại [wemoe.py:68-97](../src/wemoe_nlp/wemoe.py#L68-L97), trung bình có mask tại [wemoe.py:195-203](../src/wemoe_nlp/wemoe.py#L195-L203), chia sẻ `w` tại [wemoe.py:205-220](../src/wemoe_nlp/wemoe.py#L205-L220).

## Slide 6: Pipeline

**Ý chính:** toàn bộ quy trình gồm bốn giai đoạn, và mỗi giai đoạn cập nhật những thứ khác nhau.

| Giai đoạn | Loại | Làm gì | Cập nhật |
|---|---|---|---|
| 1. Fine-tune | Huấn luyện | Tối thiểu cross-entropy trên từng tác vụ | Một chuyên gia |
| 2. Merge | Dựng mô hình | Tính `τ_e` và Task Arithmetic | Không có |
| 3. FSFT | Huấn luyện | Tối thiểu tổng cross-entropy của 5 tác vụ | Chỉ router `φ` |
| 4. Predict | Suy luận | Tính `W(x)` rồi lấy lớp có xác suất cao nhất | Không có |

Công thức giai đoạn 3 lấy tổng trên 5 tác vụ. Mỗi bước huấn luyện lấy một batch 16 mẫu của từng tác vụ, tính loss và cộng dồn gradient, rồi cập nhật router một lần.

Bài báo gốc huấn luyện router bằng cách làm mô hình tự tin hơn trên dữ liệu test không nhãn. Đề tài dùng nhãn thật trên `k` mẫu riêng, nên cách huấn luyện khác bài báo.

**Kiểm tra:** chuyên gia tại [experts.py:134-148](../src/wemoe_nlp/experts.py#L134-L148), vòng lặp FSFT tại [fsft.py:165-180](../src/wemoe_nlp/fsft.py#L165-L180).

## Slide 7: Data boundary

**Ý chính:** các phần dữ liệu được tách riêng để tránh rò rỉ.

Tập train chính thức của mỗi tác vụ được chia một lần, với seed cố định:

| Phần | Kích thước | Dùng cho |
|---|---|---|
| `expert_train` | Phần còn lại | Fine-tune chuyên gia |
| `fewshot_pool` | 256 mẫu mỗi tác vụ | Chỉ huấn luyện router |
| `dev` | 10%, tối đa 1.000 | Chọn epoch, `λ` và bước tốt nhất của router |

Tập validation chính thức chỉ dùng để tính điểm cuối cùng.

Tại sao phải tách: nếu chuyên gia đã thấy mẫu few-shot, router sẽ học trên dữ liệu mà chuyên gia đã thuộc, và kết quả sẽ đẹp giả tạo.

Với tác vụ nhỏ, cái giá khá rõ: RTE chỉ còn 1.985 mẫu để huấn luyện chuyên gia. Điều này có thể góp phần làm chuyên gia RTE yếu (slide 14).

**Kiểm tra:** [data.py:64-78](../src/wemoe_nlp/data.py#L64-L78), [configs/default.yaml](../configs/default.yaml), mục 2.2 của [báo cáo](../docs/report/README.md).

## Slide 8: Experiments

**Ý chính:** các mô hình được xếp từ mức sàn đến mức tham chiếu để dễ so sánh.

| Mô hình | Vai trò |
|---|---|
| Majority | Mức sàn: mô hình không học được gì |
| Simple Averaging | Task Arithmetic với `λ = 1/5`, chính là trung bình các trọng số |
| Task Arithmetic | Hợp nhất tĩnh tốt nhất, `λ` chọn trên dev |
| WEMoE init | WEMoE trước khi huấn luyện router |
| WEMoE + FSFT | Phương pháp chính |
| Năm chuyên gia | Mức tham chiếu phía trên |

Cách chấm điểm khác nhau theo tác vụ. MCC dùng cho CoLA vì dữ liệu lệch lớp: mô hình đoán một lớp vẫn có accuracy cao, nhưng MCC bằng 0. Mọi điểm đều nhân 100.

**Kiểm tra:** [baselines.py:64-72](../src/wemoe_nlp/baselines.py#L64-L72), `task_score` trong [metrics.py](../src/wemoe_nlp/metrics.py).

## Slide 9: Main result

**Ý chính:** FSFT mới là thứ tạo ra cải thiện, không phải bản thân kiến trúc khi chưa huấn luyện.

Cách đọc bảng từ trên xuống:

- Task Arithmetic (57,4) và WEMoE init (57,8) gần như bằng nhau. Điều này hợp lý vì router khởi tạo xấp xỉ Task Arithmetic.
- Sau FSFT, điểm lên 69,9, nhưng vẫn dưới 78,2 của các chuyên gia.

**Gain over majority** cho biết đã lấy lại bao nhiêu phần khoảng cách từ mức đoán bừa đến chuyên gia:

```text
gain = (điểm − majority) / (chuyên gia − majority)
0%   = không hơn đoán hằng
100% = bằng chuyên gia
```

Chỉ số này tính cho từng tác vụ rồi lấy trung bình. Nó cần thiết vì mức đoán hằng đã khá cao: trên MRPC đã là 75 và trên các tác vụ accuracy khoảng 50. Theo chỉ số này, Task Arithmetic chỉ đạt 18%, còn WEMoE + FSFT đạt 59%.

Dấu `± 0,1` là độ lệch chuẩn trên ba seed của router. Chuyên gia và cách chia dữ liệu không đổi giữa các seed.

**Kiểm tra:** [results/summary.md](../results/summary.md), mục 3 của [báo cáo](../docs/report/README.md), `gain_over_majority` trong [metrics.py](../src/wemoe_nlp/metrics.py).

## Slide 10: Few-shot budget

**Ý chính:** ít nhãn thì kết quả thấp và không ổn định.

| `k` | Điểm | Nhận xét |
|---|---|---|
| 16 | 60,8 ± 2,7 | Overfit sớm, dao động lớn |
| 64 | 64,6 ± 3,7 | Tốt hơn nhưng vẫn phụ thuộc seed |
| 256 | 69,9 ± 0,1 | Ổn định |

"Best dev step" là bước huấn luyện có điểm dev cao nhất. Với `k` nhỏ, bước này đến sớm (50–250 trên 500), nghĩa là router bắt đầu học thuộc dữ liệu từ sớm.

Dòng cuối của slide giải thích một điều có vẻ lạ: từ `k=16` lên 64, điểm chỉ tăng 3,8 nhưng gain gần gấp đôi. Nguyên nhân là ở `k=16`, một phần điểm đến từ MRPC và RTE đang đoán gần như một lớp.

**Kiểm tra:** mục 4 của [báo cáo](../docs/report/README.md), `best_step` trong các file `results/wemoe/*/results.json`.

## Slide 11: Per-task results

**Ý chính:** điểm trung bình tốt không có nghĩa là tốt trên mọi tác vụ.

| Tác vụ | Majority | Full MLP | Intermediate | Chuyên gia |
|---|---|---|---|---|
| MRPC (acc) | 68,4 | **76,3** | 66,5 (dưới majority) | 82,1 |
| RTE | 52,7 | 52,0 (dưới majority) | **59,0** | 63,2 |

Hai phạm vi đạt điểm trung bình gần nhau nhưng thắng ở những tác vụ khác nhau.

Ví dụ tính gain cho RTE của full MLP: `(52,0 − 52,7) / (63,2 − 52,7) ≈ −7%`. Giá trị âm nghĩa là kém hơn cả đoán hằng.

Dòng cuối cảnh báo rằng tập validation của RTE (277 mẫu) và MRPC (408 mẫu) khá nhỏ. Sai số chuẩn khoảng 3 và 2 điểm, nên một phần chênh lệch có thể chỉ là nhiễu.

**Kiểm tra:** [results/summary.md](../results/summary.md), mục 3 và 5 của [báo cáo](../docs/report/README.md).

## Slide 12: Storage trade-off

**Ý chính:** WEMoE tiết kiệm ít hơn tưởng tượng, vì phải lưu task vector.

Mô hình full MLP lưu 400,0M tham số:

| Thành phần | Tham số |
|---|---|
| Task vector của MLP | 283,3M |
| Trọng số MLP gốc | 56,7M |
| Phần BERT dùng chung | 52,8M |
| Router | 7,1M |
| Head | Rất nhỏ |

Phần lớn dung lượng là task vector, nên bỏ task vector của fc2 giúp bản intermediate-only còn 258,4M, nhỏ hơn khoảng 35%.

**Lưu ý quan trọng nhất của slide:** vì mỗi tác vụ có head riêng, ta đã biết tác vụ khi suy luận. Nếu đã biết tác vụ, chỉ cần tải đúng một chuyên gia là xong, vừa chính xác hơn, vừa chỉ tốn thêm khoảng 37% so với full MLP. Vì vậy, lý lẽ cho WEMoE phải dựa vào bản intermediate-only hoặc các tình huống không biết tác vụ.

Thời gian trên Tesla T4 là thời gian huấn luyện, không phải tốc độ suy luận.

**Kiểm tra:** `parameter_breakdown` tại [wemoe.py:343-361](../src/wemoe_nlp/wemoe.py#L343-L361), `parameters` trong `results/wemoe/full-mean-k256-s0/results.json`, mục 7 của [báo cáo](../docs/report/README.md).

## Slide 13: Routing

**Ý chính:** nhìn vào router để hiểu vì sao RTE thất bại.

Cách đọc biểu đồ:

- Mỗi ô lớn là một loại đầu vào, ví dụ các câu của RTE.
- Hàng là tầng BERT, cột là task vector của từng chuyên gia.
- Màu xanh là cộng task vector, màu đỏ là trừ.

Ba quan sát:

1. **Mẫu hình khá giống nhau giữa các tác vụ:** tương quan trung bình 0,82. Router không đơn thuần chọn đúng chuyên gia của tác vụ.
2. **Vẫn có phụ thuộc đầu vào:** loại tác vụ giải thích khoảng 58% phương sai.
3. **Hệ số vượt xa `λ`:** trải từ −3,0 đến 4,4, tức router ngoại suy chứ không chỉ trộn.

Giải thích cho RTE:

| Trường hợp | Hệ số task vector RTE trên câu RTE | Điểm RTE |
|---|---|---|
| Full MLP | Khoảng 0,01, và −1,2 ở tầng 11–12 | 52,0 |
| Intermediate-only | 0,96 | 59,0 |

Để so sánh, MRPC tự cho chuyên gia của mình hệ số 1,37.

Ghi chú nhỏ cuối slide rất quan trọng: thí nghiệm dùng hệ số không phụ thuộc đầu vào (`router_hidden_layers=0`) chưa có kết quả. Vì vậy, biểu đồ **chưa chứng minh** rằng phụ thuộc đầu vào là nguyên nhân của mức tăng điểm. Có thể chỉ cần học hệ số tốt hơn cho từng tầng là đủ.

**Kiểm tra:** mục 6 của [báo cáo](../docs/report/README.md). Các số này chưa tính lại được, vì `predictions_validation.jsonl` không có trong repo.

## Slide 14: Limitations

**Ý chính:** nêu rõ những gì kết quả chưa chứng minh được.

- **RTE:** vẫn dưới majority, và bản thân chuyên gia RTE cũng yếu (63,2), nên mô hình hợp nhất có ít thứ để học theo.
- **`k` nhỏ dao động:** độ lệch chuẩn của CoLA MCC tới 14,1 ở `k=16`.
- **Dev dùng nhiều nhãn hơn FSFT:** router học trên 256 mẫu mỗi tác vụ nhưng chọn bước dừng bằng tối đa 1.000 mẫu, nên "few-shot" chưa thật chặt chẽ.
- **Ít seed:** seed chỉ thay đổi router và mẫu few-shot, không thay đổi chuyên gia hay cách chia dữ liệu.

Câu cuối nhắc rằng so sánh với bài báo chỉ mang tính định tính, vì khác dữ liệu và cách huấn luyện router.

**Kiểm tra:** mục 9 và 10 của [báo cáo](../docs/report/README.md).

## Slide 15: Conclusion

Trả lời câu hỏi ở slide 2:

- **Có lấy lại được chất lượng không?** Một phần: 57,4 lên 69,9, trong khi chuyên gia đạt 78,2.
- **Có tốt hơn trên mọi tác vụ không?** Không: full MLP vẫn dưới majority ở RTE.
- **Việc tiếp theo:** thêm seed, few-shot chặt chẽ hơn, thí nghiệm routing không phụ thuộc đầu vào và độ trễ suy luận.

---

# Phần dự phòng (slide 16–24)

Các slide này chỉ dùng khi có câu hỏi. Slide 16, 17 và phần MoE cổ điển ở slide 20 là kiến thức nền từ các bài báo gốc, không phải kết quả của đề tài.

## Slide 16: Transformer

Kiến trúc gốc gồm encoder và decoder. Đề tài chỉ dùng encoder thông qua BERT.

- **Self-attention:** mỗi token tính độ liên quan với các token khác, rồi tổng hợp thông tin theo độ liên quan đó. Chia cho `√d_k` để giá trị không quá lớn trước khi qua softmax.
- **Multi-head attention:** chạy nhiều attention song song để nắm bắt nhiều kiểu quan hệ.
- **Positional encoding:** attention tự nó không biết thứ tự từ, nên cần cộng thêm thông tin vị trí.
- **FFN và residual:** FFN chính là khối "MLP" mà WEMoE nâng cấp. Residual cộng đầu vào vào đầu ra để dễ huấn luyện.

## Slide 17: BERT

BERT chỉ dùng encoder và đọc ngữ cảnh hai chiều.

- **Đầu vào:** `[CLS] câu A [SEP] câu B [SEP]`. Tác vụ một câu thì không có câu B.
- **Embedding:** tổng của token, segment (câu A hay B) và position. Position của BERT được học, khác với dạng hình sin ở slide 16.
- **Pre-training:** MLM đoán từ bị che, NSP đoán câu B có đứng sau câu A không. Hai head này không được giữ lại.

Đề tài tải `bert-base-uncased` với head phân loại 2 lớp mới.

**Kiểm tra:** [experts.py:56-61](../src/wemoe_nlp/experts.py#L56-L61).

## Slide 18: BERT fine-tuning

Bảng trên liệt kê 5 tác vụ và điểm của chuyên gia. Điểm 85,1 của MRPC là trung bình của accuracy 82,1 và F1 88,1.

Bảng dưới phân biệt hai loại trọng số:

| | `θ_0` (pretrained) | `θ_e` (chuyên gia) |
|---|---|---|
| Vai trò trong WEMoE | Trọng số chung | Tạo ra `τ_e` và head |
| Cập nhật khi fine-tune | Là điểm xuất phát | Toàn bộ BERT và head |

Mọi tác vụ dùng chung một công thức huấn luyện. Vì cả năm đều là phân loại hai lớp nên không có head hồi quy.

**Kiểm tra:** mục `expert` trong [configs/default.yaml](../configs/default.yaml), `results/baselines.json`.

## Slide 19: Weight ensembling và logit ensembling

**Logit ensembling** chạy từng mô hình rồi trộn đầu ra. **Weight ensembling** trộn trọng số trước rồi chỉ chạy một mô hình.

Logit ensembling chỉ được nêu để so sánh về mặt khái niệm. Nó không được cài đặt hay đo trong repo, nên không có con số so sánh hiệu năng.

Câu cuối slide là ý dễ bị hỏi. Với **một** lớp tuyến tính, hai cách là tương đương về đại số:

```text
W(x)·h = W_0·h + Σ_e w_e · (τ_e·h)
```

Code có chế độ `decomposed` tính theo đúng công thức này. Nhưng vì có GELU và 12 tầng xếp chồng, toàn bộ mô hình không còn tương đương với việc trộn đầu ra.

**Kiểm tra:** `WeightEnsembledLinear` tại [wemoe.py:100-158](../src/wemoe_nlp/wemoe.py#L100-L158).

## Slide 20: Router và Mixture of Experts

MoE cổ điển trộn **đầu ra** của các chuyên gia đang chạy. WEMoE trộn **trọng số**, với các chuyên gia là task vector đã đóng băng.

- **Sparse:** chỉ một vài chuyên gia hoạt động, ví dụ top-k.
- **Dense:** mọi chuyên gia đều tham gia. WEMoE trong đề tài là dense: cả 5 task vector ở mọi tầng.
- **Không phải tổ hợp lồi:** hệ số có thể âm và tổng không phải bằng 1.

## Slide 21: Task metrics

Công thức điểm nằm ở slide 8. Điểm trung bình là trung bình không trọng số của năm tác vụ, nên tác vụ nhỏ như RTE có sức nặng bằng tác vụ lớn như QNLI.

**Kiểm tra:** `task_score`, `mean_score` trong [metrics.py](../src/wemoe_nlp/metrics.py).

## Slide 22: Router training

Tóm tắt cấu hình FSFT: AdamW, lr `10⁻⁴`, batch 16 mỗi tác vụ, tối đa 500 bước, đánh giá trên dev mỗi 50 bước và giữ lại router tốt nhất.

**Kiểm tra:** mục `wemoe` trong [configs/default.yaml](../configs/default.yaml), [fsft.py:146-148](../src/wemoe_nlp/fsft.py#L146-L148), [fsft.py:184-191](../src/wemoe_nlp/fsft.py#L184-L191).

## Slide 23: Bài báo WEMoE và đề tài

Bảng so sánh các điểm khác nhau chính. Hai điểm quan trọng nhất là dữ liệu (ảnh và văn bản) và cách huấn luyện router (entropy không nhãn và cross-entropy có nhãn). Bài báo không có kết quả trên GLUE, nên không thể so sánh trực tiếp các con số.

**Kiểm tra:** bảng "Differences from the paper" trong [README.md](../README.md), mục 10 của [báo cáo](../docs/report/README.md).

## Slide 24: References

Danh sách 5 tài liệu được trích dẫn trong slide.

---

# Tự kiểm tra bằng code

Các lệnh sau chạy từ thư mục gốc `wemoe-nlp`:

```bash
python -m pytest -q                  # kiểm tra đơn vị, chạy trên CPU với BERT nhỏ
python scripts/summarize_results.py  # tạo lại results/summary.md
```

Huấn luyện thật cần GPU. Xem [docs/guide/README.md](../docs/guide/README.md) để chạy trên Colab.
