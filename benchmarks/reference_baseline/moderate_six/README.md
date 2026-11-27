# Baseline chung: 6 instance Moderate

Bộ kết quả MOEA/D cơ bản để các thành viên dùng chung khi so sánh module mới.
Phạm vi: `Mk01`–`Mk03` của Brandimate1993 và `01a`–`03a` của
DauzerePeresPaulli1994, đều ở mức hạn chót `Moderate`.

- 10 seed chính (0–9) cho mỗi instance: 60 lượt, mỗi lượt 1.000 NFE.
- Lặp seed 0 một lần cho mỗi instance: 6 lượt kiểm tra tái lập, không tính vào HV trung bình.
- 60/60 lượt chính hợp lệ; 6/6 lượt lặp khớp trace, archive và lịch.
- Mã tạo kết quả: commit `6b3b1fa` trên nhánh `greywolf`; hash mã và input chi tiết nằm trong `manifest.json`.

| Đường dẫn | Ý nghĩa |
| --- | --- |
| `runs/` | 60 JSON gốc: nghiệm archive, mục tiêu, gene, lịch, trace và checkpoint từng seed. Dùng để tính IGD+ và Coverage về sau. |
| `manifest.json` | Protocol, sáu instance, cận chuẩn hóa, hash mã và dữ liệu. |
| `completion.json`, `replays/` | Trạng thái hoàn tất và sáu lượt chạy kiểm tra tái lập. |
| `summary.csv`, `summary.md`, `summary.json`, `runs.csv` | Bảng HV và thống kê để xem nhanh; không thay cho `runs/`. |

Khi đã có study mới trên **cùng sáu instance** (cùng ID và cận chuẩn hóa),
có thể so sánh bằng lệnh sau từ gốc repo:

```powershell
./.venv/Scripts/python.exe scripts/compare_studies.py --study-a benchmarks/reference_baseline/moderate_six --study-b results/my_study --output results/compare_to_baseline
```

Script đọc trực tiếp `manifest.json` và `runs/`; không cần chạy lại baseline.
Bộ này chỉ đại diện cho sáu instance trên, không phải toàn bộ 84 instance.