# MOEA/D cho FJSP

Project giải FJSP tĩnh với minimize cả ba mục tiêu: makespan, tổng thời gian
gia công và tải gia công lớn nhất trên một máy. Baseline dùng MOEA/D cơ bản,
pp phân rã Tchebycheff, khởi tạo ngẫu nhiên và toán tử OS/MS có xác suất cố định.

Tài liệu cần đọc:

Hướng dẫn chung: **[README.md](README.md)**

Giới thiệu bài toán và cấu hình **[FJSP.md](docs/FJSP.md)**

Thực nghiệm: **[EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

Dữ liệu: **[BENCHMARKS.md](docs/BENCHMARKS.md)**.
## 1. Cài đặt

Mở PowerShell tại thư mục gốc project. Cần tối thiểu Python 3.11 để dùng bộ dependency hiện tại.

```powershell
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install pip==25.3
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.venv/Scripts/python.exe -m pip install --no-index --no-build-isolation --no-deps -e ./vendor
./.venv/Scripts/python.exe -m pip check
```

Nếu đã có môi trường đúng phiên bản, bỏ qua lệnh tạo `.venv`. Các lệnh gọi trực
tiếp Python của môi trường, không cần activate.
Cài editable để chương trình dùng mã trong `vendor/`, giữ dependency theo lock.

## 2. Triển khai
Chọn data và địa chỉ lưu gene/lịch, thay đổi các args khác nếu cần:

```powershell
./.venv/Scripts/python.exe scripts/run_fjsp.py --instance instance/Brandimarte1993/mk01.txt --seed 1 --evaluations 1000 --output results/mk01_seed1.json
```

File output phải chưa tồn tại. Cách chạy nhiều seed, tiếp tục khi gián đoạn và
đọc kết quả nằm trong [EXPERIMENTS.md](docs/EXPERIMENTS.md).


Ví dụ luồng chạy và tham số mặc định:
```powershell
./.venv/Scripts/python.exe scripts/run_fjsp.py
```

Mặc định: `fattahi1`, seed 1, 91 nghiệm, 10 láng giềng, 1.000 NFE,
`pc=0.8`, `pm=0.05`. Output cần báo `actual_evaluations: 1000` và
`all_schedules_valid: true`, pareto front của ví dụ này là `(66,127,66)` và `(91,115,91)`.

## 3. Kiểm tra
Nên chạy test luồng sau khi thay đổi module và trước khi thực nghiệm để kiểm tra tính đúng/xung đột của chương trình.
```powershell
./.venv/Scripts/python.exe -m pytest vendor/moead_framework/test -q --tb=short
```

Nếu cần, có thể Kiểm tra tính đúng decoder bằng
`scripts/smoke_fjsp.py --show-operations`,
chi tiết nằm trong [FJSP.md](docs/FJSP.md).

## 4. Cấu trúc folder và nơi phát triển từng module của idea.

```text
vendor/moead_framework/
  algorithm/       # Vòng lặp MOEA/D và hàm lắp cấu hình FJSP
  core/            # Chọn cha mẹ, toán tử, sinh con, chọn bài toán con, dừng
  problem/fjsp/    # Instance, OS/MS, decoder, đánh giá và kiểm tra lịch
  aggregation/     # Hàm phân rã
  solution/        # Đối tượng nghiệm
  tool/            # Pareto, seed, HV và thu thập thực nghiệm
  test/            # Test component, FJSP và thực nghiệm
scripts/           # Entry point chạy và scripts/configs/ chứa protocol JSON
instance/          # 400 instance FJSP
docs/              # Mô hình, hướng dẫn mở rộng, thực nghiệm và tra cứu
```

Sửa mã trong `vendor/` và đặt test trong `vendor/moead_framework/test/`.
Bảng điểm mở rộng, giao diện component và quy tắc NFE/archive nằm trong
[FJSP.md](docs/FJSP.md). Thử từng module bằng cùng input, seed và ngân sách của
[protocol thực nghiệm](docs/EXPERIMENTS.md).

`results/` tạm thời đang được đặt trong gitignore.
