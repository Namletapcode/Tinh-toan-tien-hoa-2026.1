# MOEA/D cho FJSP

Project giải FJSP có hạn chót với minimize hai mục tiêu: makespan và tổng độ
trễ có trọng số (TWT). Baseline dùng MOEA/D cơ bản, phân rã Tchebycheff, khởi
tạo ngẫu nhiên và toán tử OS/MS có xác suất cố định.

Tài liệu cần đọc:

Hướng dẫn chung: **[README.md](README.md)**

Giới thiệu bài toán và cấu hình **[FJSP.md](docs/FJSP.md)**

Thực nghiệm: **[EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

Dữ liệu: **[BENCHMARKS.md](docs/BENCHMARKS.md)**.

Đánh giá mục tiêu: **[EVALUATE.md](docs/EVALUATE.md)**.
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
./.venv/Scripts/python.exe scripts/run_fjsp.py --instance "instance/main/Brandimate1993/Loose/Mk01/Mk01.fjs" --seed 1 --evaluations 1000 --output results/mk01_seed1.json
```

File output phải chưa tồn tại. Cách chạy nhiều seed, tiếp tục khi gián đoạn và
đọc kết quả nằm trong [EXPERIMENTS.md](docs/EXPERIMENTS.md).

Ví dụ luồng chạy và tham số mặc định:
```powershell
./.venv/Scripts/python.exe scripts/run_fjsp.py
```

Mặc định: Brandimate Loose `Mk01`, seed 1, 13 nghiệm (`H=12`), 10 láng giềng,
1.000 NFE, `pc=0.8`, `pm=0.05`. Output cần báo `actual_evaluations: 1000`,
`objective_names: ["makespan", "twt"]` và `all_schedules_valid: true`.

Chạy batch mặc định cả 2 bộ data (84 instance × 10 seed):

```powershell
./.venv/Scripts/python.exe scripts/run_baseline_validation.py --output-dir results/baseline_main
```

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
instance/main/  # 84 file .fjs Brandimate + Dauzere (Loose/Moderate/Tight)
docs/                 # Mô hình, hướng dẫn mở rộng, thực nghiệm và tra cứu
```

Sửa mã trong `vendor/` và đặt test trong `vendor/moead_framework/test/`.
Bảng điểm mở rộng, giao diện component và quy tắc NFE/archive nằm trong
[FJSP.md](docs/FJSP.md). Thử từng module bằng cùng input, seed và ngân sách của
[protocol thực nghiệm](docs/EXPERIMENTS.md).

`results/` tạm thời đang được đặt trong gitignore.

## 5. Micropopulation và Grey Wolf

Ba chế độ: `baseline` (mặc định), `micro_random` (đối chứng cùng số cá thể), `micro_gwo` (chọn cha mẹ theo Grey Wolf). Cách chạy, sơ đồ lớp và quy tắc nằm trong [MICRO_GWO](docs/MICRO_GWO.md).

```powershell
./.venv/Scripts/python.exe scripts/run_fjsp.py --variant micro_gwo --micro-population-size 4
```

Baseline cố định của sáu instance `Moderate` nằm trong [benchmark dùng chung](benchmarks/reference_baseline/moderate_six/README.md). Các file JSON gốc trong `runs/` phục vụ đối chiếu IGD+ và Coverage về sau.
