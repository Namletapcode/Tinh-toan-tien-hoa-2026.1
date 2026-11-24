# Micropopulation và Grey Wolf cho FJSP

[README](../README.md) · [Mô hình FJSP](FJSP.md) · [Thực nghiệm](EXPERIMENTS.md)

## 1. Ba chế độ chạy

| `variant` | Quần thể | Chọn cha mẹ |
| --- | --- | --- |
| `baseline` | Một nghiệm mỗi trọng số | Hai nghiệm ngẫu nhiên theo baseline |
| `micro_random` | Nhiều nghiệm mỗi trọng số | Hai vị trí khác nhau ngẫu nhiên trong các nhóm láng giềng |
| `micro_gwo` | Như `micro_random` | Một sói omega trong nhóm hiện tại lai với leader hoặc omega khác |

Mục tiêu vẫn là `(makespan, twt)`, cùng minimize. Decoder, OS/MS, POX, lai đều
MS và đột biến giữ nguyên. Không thêm worker, setup, vận chuyển hoặc Q-learning.

`N = H + 1` là số trọng số/nhóm. `mu` là số cá thể mỗi nhóm, tối thiểu 4.
`PS = N * mu` là tổng cá thể và số đánh giá khởi tạo. Mặc định `H=12`,
`mu=4`: 13 nhóm, 52 cá thể, 10 nhóm láng giềng. Baseline mặc định vẫn có 13
cá thể. Không dùng `PS` thay cho số trọng số hay giới hạn láng giềng.

## 2. Nơi đọc và phát triển mã

| File trong `vendor/moead_framework/` | Trách nhiệm |
| --- | --- |
| `algorithm/fjsp.py` | Factory baseline gốc, giữ nguyên |
| `algorithm/fjsp_variants.py` | Tên chế độ, kích thước quần thể, factory chung cho runner |
| `algorithm/moead_micro_gwo.py` | Lớp `MoeadMicroGWO`, giữ nhóm, khởi tạo, cập nhật nhóm và đại diện |
| `core/parent_selector/micro_gwo_selector.py` | Xếp hạng leader, tính tiến độ/Bf và chọn cha mẹ |
| `core/offspring_generator/fjsp.py` | Sao chép gene cha mẹ, gọi toán tử, đánh giá một con |
| `test/test_micro_gwo.py` | Kiểm tra ngân sách, nhóm, chọn cha mẹ, archive, replay và runner |

`MoeadMicroGWO` kế thừa `Moead`. Vòng lặp trong `AbstractMoead` vẫn sinh đúng
một con mỗi lần thăm một bài toán con. Lớp mới override phần khởi tạo, `init_z`
và `update_solutions`. Hai chế độ micro dùng cùng lớp, khác `parent_selection`.

Trạng thái chính là `micro_populations[group][slot]`. `population[group]` chỉ
là **bản sao** nghiệm đại diện tốt nhất theo trọng số nhóm và `z` hiện tại,
để các giao diện cũ tiếp tục hoạt động. Kiểm tra cuối lượt phải dùng
`population_solutions()` để xét mọi cá thể, không chỉ đại diện.
Module không sửa trực tiếp cá thể đang lưu trong nhóm. Khi nhận nghiệm mới,
thay bằng đối tượng mới. Bản sao đại diện được dùng lại khi cá thể nguồn không
đổi, nhưng vẫn xếp hạng lại theo `z` sau mỗi con.

## 3. Một lần sinh con

1. Gom nhóm hiện tại và các nhóm láng giềng. Chấm mọi ứng viên theo **cùng
   trọng số nhóm đang tối ưu**, cùng `z`. Hòa điểm thì dùng chỉ số nhóm/vị trí.
2. Chọn tối đa ba chromosome khác nhau có điểm nhỏ nhất làm alpha, beta, delta.
   Nếu bài quá nhỏ chỉ có một hoặc hai chromosome khác nhau, dùng số leader có
   sẵn, không đánh giá lại để cố tìm đủ ba.
3. Các vị trí ngoài leader là omega. Chọn một omega trong nhóm hiện tại bằng
   con trỏ luân phiên. Với `mu >= 4`, luôn còn ít nhất một vị trí như vậy.
4. Tính `progress = (current_eval - PS) / (max_evaluation - PS)` và
   `Bf = (1 - progress) * B0`. Lấy `B0` mới mỗi con.
5. Nếu `Bf >= 0.5`, chọn ngẫu nhiên một leader để lai. Ngược lại, chọn omega
   khác trong pool, ưu tiên gene khác. Nếu chỉ còn một vị trí omega, dùng vị trí
   khác trong pool làm dự phòng. Không sửa cha mẹ hay đánh giá trong selector.
6. Bộ sinh con hiện có sao chép gene, lai/đột biến, rồi gọi `evaluate()` một lần.
7. Vòng lặp tăng một NFE, cập nhật `z`, cập nhật nhóm và archive.

Tại nhóm láng giềng j: nếu gene con chưa tồn tại và điểm theo w_j **nhỏ hơn**
điểm của cá thể tệ nhất, thay đúng một vị trí bằng bản sao con. Mỗi nhóm có thể
nhận con độc lập. Điểm bằng nhau không thay. Khởi tạo ngẫu nhiên có thể có
chromosome trùng, không tiêu thêm NFE để ép đủ cá thể khác nhau. Sau cập nhật,
chọn lại mọi đại diện vì `z` mới có thể đổi thứ hạng.

Archive giữ nghiệm không trội từ **mọi đánh giá**, gồm cả khởi tạo và con bị
tất cả nhóm từ chối. Chấm điểm từ F đã lưu, sao chép và chọn leader không tốn NFE.
Nếu ngân sách đúng bằng PS, chỉ khởi tạo, không gọi selector. Nếu nhỏ hơn PS,
báo lỗi trước đánh giá đầu tiên. Module sinh nhiều ứng viên phải có điều phối
NFE riêng, không được giấu đánh giá phụ trong selector/toán tử.

## 4. Cách chạy

PowerShell tại gốc repo, môi trường cài theo README:

```powershell
./.venv/Scripts/python.exe scripts/run_fjsp.py --variant micro_gwo --micro-population-size 4 --output results/micro_gwo_seed1.json
./.venv/Scripts/python.exe scripts/run_fjsp.py --variant micro_random --micro-population-size 4 --output results/micro_random_seed1.json
```

File output cần chưa tồn tại. JSON có `variant`, `number_of_subproblems`,
`micro_population_size`, `population_size`, `initial_evaluations`, số lần chọn
nhánh leader/peer/random và gene/F của từng nhóm. Archive vẫn có lịch đầy đủ.

```powershell
./.venv/Scripts/python.exe scripts/run_baseline_validation.py --config scripts/configs/micro_gwo_validation.json --output-dir results/micro_gwo
./.venv/Scripts/python.exe scripts/run_baseline_validation.py --config scripts/configs/micro_random_validation.json --output-dir results/micro_random
```

Hai protocol mới chạy đủ 84 instance, mỗi instance 10 seed và lặp seed 0,
ngân sách 1.000 NFE/lượt. Checkpoint: 52, 250, 500, 1.000. Thêm `--resume`
để tiếp tục khi code/input/protocol/môi trường khớp, hoặc `--no-plot` để bỏ hình.
Không đổi protocol baseline gốc. Mỗi biến thể cần thư mục kết quả riêng.

## 5. Đối chiếu và giả định từ paper

Nguồn: Qiu, Guo & Guo, *An improved MOEA/D algorithm for solving the dual-resource
constrained flexible job shop scheduling problem*, AIITA 2025, mục III-A/III-D,
phương trình (23), DOI: https://doi.org/10.1109/AIITA65135.2025.11047607.

Đây là bản thích nghi cho FJSP makespan/TWT, chưa phải tái lập đầy đủ Q-MOEA/D.
Những lựa chọn cần ghi khi báo cáo:

- `mu=4`, cách lấy leader từ các nhóm lân cận, xử lý hòa và thiếu leader.
- Hiểu nhánh lai đầu tiên chưa rõ trong paper là omega-alpha. Ba leader có cùng
  xác suất chọn, độc lập với xác suất thực hiện lai `pc=0.8` của toán tử.
- Lấy B0 mới mỗi con và thay t/MG bằng tiến độ NFE sau khởi tạo. Do công thức,
  nửa sau ngân sách tìm kiếm chỉ còn nhánh peer. Không đảo nhánh này.
- Giữ toán tử OS/MS và đột biến baseline, quy tắc thay cá thể tệ nhất và bỏ
  trùng gene trong nhóm là quyết định tích hợp, không phải chi tiết paper đã nêu.

So sánh `baseline`, `micro_random`, `micro_gwo` trên cùng input/seed/NFE/cách đo.
So `micro_random` với `micro_gwo` để tách hiệu quả chọn cha mẹ khỏi hiệu quả
tăng quần thể. Ở 1.000 NFE, baseline còn 987 lượt sinh con, micro còn 948.
So đường hội tụ tại các mốc chung sau khởi tạo (ví dụ 52, 250, 500, 1.000),
không so HV tại NFE 13 với NFE 52 như cùng một ngân sách.

`scripts/compare_studies.py` dùng được với output mới. Manifest giữ protocol
và hash toàn bộ mã runtime, bao gồm module mới. Khi ghép Q-learning sau này,
phải tính cả mọi đánh giá khởi tạo phụ vào NFE và giữ biến thể không Q-learning
làm đối chứng. Chưa thêm cơ chế tự retry để ép nghiệm khác nhau.
