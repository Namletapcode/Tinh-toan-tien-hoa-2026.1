# Thực nghiệm và đối chiếu module

## 1. Cấu hình baseline

Luồng thí nghiệm mặc định đọc cấu hình tham số từ [baseline_validation.json](../scripts/configs/baseline_validation.json).
Mỗi đợt lưu bản protocol và hash trong manifest. Đây là cấu hình cố định để
đối chiếu, không phải bộ tham số đã tinh chỉnh tốt nhất cho mọi bài.

| Thành phần | Giá trị |
| --- | --- |
| Thuật toán / phân rã | `Moead` cơ bản / Tchebycheff|
| Khởi tạo | Ngẫu nhiên |
| Trọng số / quần thể | Lưới simplex ba chiều `H=12` / 91 nghiệm |
| Láng giềng | 10, theo khoảng cách giữa các trọng số |
| `pc` / `pm` | 0.8 / 0.05, cố định, cách áp dụng xem [FJSP.md](FJSP.md) |
| Ngân sách | 1.000 NFE/lượt, gồm 91 nghiệm đầu |
| Seed chính / lặp độc lập | 0–9 / chạy lại seed 0 |
| Checkpoint | 91, 250, 500, 1.000 NFE |
| Tập bài | 9 instance thuộc 7 họ dữ liệu, liệt kê trong bảng đối chiếu |

Quy ước NFE, archive và đánh giá phụ khi mở rộng nằm trong [FJSP.md](FJSP.md).

## 2. Run, resume và lưu kết quả

Từ thư mục gốc, chọn một thư mục output chưa tồn tại:

```powershell
./.venv/Scripts/python.exe scripts/run_baseline_validation.py --config scripts/configs/baseline_validation.json --output-dir results/my_experiment
```

Protocol trên chạy 90 lượt chính và 9 lượt lặp seed, tổng 99.000 NFE.
Để tiếp tục đợt bị gián đoạn hoặc dựng lại bảng từ các lượt đã lưu:

```powershell
./.venv/Scripts/python.exe scripts/run_baseline_validation.py --config scripts/configs/baseline_validation.json --output-dir results/my_experiment --resume
```

`--resume` kiểm tra checksum và tái sử dụng các lượt đã hoàn tất, chỉ chạy
phần thiếu. Mã, input, protocol, Python và NumPy phải khớp manifest, nếu thay
đổi, dùng thư mục mới. Không sửa raw để vượt kiểm tra. `--no-plot` bỏ xuất hình.
Luôn truyền `--output-dir` rõ ràng để chọn đúng đợt.

Cấu trúc folder từng lượt kết quả trong **folder results/** :
| Artifact | Nội dung |
| --- | --- |
| `manifest.json` | Protocol, hash mã/input, môi trường và cận chuẩn hóa |
| `runs/` | JSON từng lượt chính: trace đủ NFE, checkpoint, gene và lịch archive |
| `replays/` | JSON lượt lặp seed |
| `runs.csv` | Chỉ số của từng lượt chính |
| `summary.csv/json/md` | Tổng hợp theo instance |
| `convergence.png` | HV trung bình và dải ± độ lệch chuẩn mẫu tại các checkpoint |
| `completion.json` | Ghi nhận hoàn tất các lượt và kiểm tra |

Raw đã tồn tại không bị ghi đè, bảng/hình tổng hợp có thể dựng lại.
`results/` được bỏ qua trong Git.

## 3. Điều kiện hợp lệ một lượt chạy

Một lượt được chấp nhận khi: NFE thuật toán = NFE bài toán = độ dài trace =
ngân sách, mọi lịch đánh giá khả thi, quần thể/archive cuối có gene, mục tiêu
và lịch khớp nhau. Tại mỗi checkpoint, archive phải bằng tập không trội tính
độc lập từ toàn bộ trace đến mốc đó và không trùng vector mục tiêu. Lặp seed
so sánh trace, gene/lịch archive và checkpoint, không so thời gian.

Đánh giá HV (Hypervolume):

HV được tính trên 3 mục tiêu đều cần tối thiểu hóa (minimize). Các giá trị chỉ được chuẩn hóa khi báo cáo:

```text
f_scaled[i] = (f[i] - L[i]) / max(U[i] - L[i], 1)
HV_score = HV(front_scaled, reference=(1.1, 1.1, 1.1)) / 1.1^3
```

- L/U: Cận dưới/trên được tính trước từ dữ liệu đầu vào, không phụ thuộc kết quả chạy. Cận dựa trên thời gian xử lý, đường công việc và tải máy; riêng cận makespan giả định decoder không chủ động chờ.

- Xử lý giá trị: Mục tiêu cố định được chuẩn hóa về 0; giá trị vượt cận bị từ chối, không cắt ngưỡng.

- Ý nghĩa HV: HV càng lớn càng tốt khi so sánh trên cùng instance, cận L/U và điểm tham chiếu. HV không phải phần trăm đạt tối ưu.

- HV archive: Không giảm chỉ chứng tỏ archive lưu giữ nghiệm, chưa đủ kết luận thuật toán cải thiện.

- IGD: Chưa tính cho bài toán lớn vì thiếu Pareto front tham chiếu.

- Giá trị cực tiểu: Min của từng mục tiêu có thể đến từ các nghiệm hoặc seed khác nhau, không được ghép thành một nghiệm chung.

- Chi tiết cách tính cận nằm trong [objective_bounds](../vendor/moead_framework/tool/fjsp_experiments.py).

Thời gian một lượt chạy gồm: khởi tạo, tìm kiếm, kiểm tra khả thi trong đánh giá và thu
checkpoint. Không gồm đọc input, tính HV/Pareto độc lập, kiểm tra sau chạy hoặc
xuất file. So thời gian cần cùng máy, môi trường và cách kiểm tra.

## 4. Mốc kết quả đối chiếu
Kết quả lượt chạy baseline. HV dưới đây là trung bình ± độ lệch
chuẩn mẫu trên 10 seed tại NFE 1.000, Cmax là giá trị tốt nhất quan sát được.

| Instance | HV trung bình ± SD | Cmax tốt nhất |
| --- | ---: | ---: |
| fattahi1 | 0.905939 ± 0.000000 | 66 |
| kacem1 | 0.944877 ± 0.025330 | 11 |
| mk01 | 0.812470 ± 0.043901 | 42 |
| behnke1 | 0.707009 ± 0.033969 | 125 |
| dpp01a | 0.951473 ± 0.021070 | 2786 |
| mt10c1 | 0.917214 ± 0.006485 | 1005 |
| hurink_sdata_ft06 | 0.948485 ± 0.006546 | 55 |
| hurink_rdata_ft06 | 0.927225 ± 0.015282 | 50 |
| hurink_sdata_orb7 | 0.932622 ± 0.004150 | 426 |

Nhận diện mốc bằng fingerprint, không dùng các con số trên để suy ra tối ưu:

```text
code_sha256     f121cfcba2f4bacde18291d0b030707e709ed3dffb0e9377f8c61acaf31f52c9
protocol_sha256 bcd3335601bd495cbc4140b3f254d46a3a999b323fbb1623e0b9071664e75fad
```

Fingerprint mã gồm Python runtime, hai entrypoint chạy và file lock, docs/test
không nằm trong fingerprint này. Input có hash riêng trong manifest. Đây không
phải mã commit Git. Bộ artifact tham chiếu được lưu cục bộ tại
`results/baseline_current_20261113/`, cần chuyển riêng khi chia sẻ qua Git.

## 5. Tích hợp idea module

Khi thử module: chọn rõ một thành phần, giữ input/seed/trọng số/ngân sách và
cách đo như baseline, ghi mọi thay đổi có chủ đích. Chạy test component cùng
bộ test chung trước thực nghiệm. Tính mọi đánh giá phụ vào NFE. Lưu cấu hình
module và nguồn mã đã dùng cùng artifact để truy ngược kết quả. So sánh theo
từng instance trên cùng danh sách seed, báo cả trung bình và độ phân tán.
