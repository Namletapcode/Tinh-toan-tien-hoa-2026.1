# Khởi tạo quần thể bằng Q-learning

[README](../README.md) · [Thực nghiệm](EXPERIMENTS.md) · [Mô hình FJSP](FJSP.md)

Module thay **duy nhất** bước khởi tạo của `Moead` (`MoeadQInit.initial_population`).

## 1. Các chế độ

| `initializer` | Mô tả |
| --- | --- |
| `baseline` | `Moead` gốc, khởi tạo ngẫu nhiên (mốc so sánh, mặc định) |
| `qlearning` | Toàn bộ 13 nghiệm đầu do Q-learning xây dựng |
| `hybrid_q_random` | `random_ratio` nghiệm ngẫu nhiên (giữ đa dạng), phần còn lại do Q-learning |

## 2. Mô hình MDP

Mỗi **episode** xây dựng một chromosome OS/MS từng thao tác một, nhắm vào một
subproblem có trọng số `λ = (λ_Cmax, λ_TWT)`. **Một episode = một nghiệm = 1 NFE.**

- **State** `s = (bin trọng số, bin tiến độ, bin áp lực trễ)`, mặc định 3 × 4 × 3 = 36 trạng thái.
  - bin trọng số: `λ_Cmax` chia 3 khoảng, để một bảng Q học chính sách khác nhau
    cho subproblem thiên về makespan và thiên về TWT;
  - tiến độ: tỉ lệ thao tác đã xếp;
  - áp lực trễ: tỉ lệ job chưa xong có `slack = d_j − ready_j − work_còn_lại < 0` (0 / ≤ 50% / > 50%).
- **Action**: luật dispatching ghép (luật chọn job × luật chọn máy), 8 action.

  | Luật job | Ý nghĩa | | Luật máy | Ý nghĩa |
  | --- | --- | --- | --- | --- |
  | MWKR | job còn nhiều việc nhất | | ECT | máy hoàn thành thao tác sớm nhất (có xét tải) |
  | WSLACK | slack có trọng số nhỏ nhất (ưu tiên job nặng đã/ sắp trễ) | | SPT | máy có thời gian xử lý ngắn nhất |
  | EEC | job có thao tác kế tiếp hoàn thành sớm nhất | | | |
  | RANDOM | job hợp lệ ngẫu nhiên | | | |

  Mọi action chỉ chọn trong các job còn thao tác và các máy hợp lệ, nên chromosome luôn khả thi.
- **Reward** mỗi bước: `r_t = −(λ_Cmax · ΔCmax_t / s_C + λ_TWT · ΔTWT_t / s_T)`,
  với `s_C` là cận dưới makespan tính từ dữ liệu và `s_T = Σw_j · s_C`.
  `ΔTWT_t` phát sinh khi thao tác cuối của một job được xếp. Các số gia cộng dồn
  đúng bằng mục tiêu cuối, nên **tổng reward không chiết khấu = −(mục tiêu vô hướng hóa theo λ)**.
- **Cập nhật** Q-learning off-policy:
  `Q(s,a) ← Q(s,a) + α [r + γ · max_a' Q(s',a') − Q(s,a)]`, thực hiện ngược từ bước cuối
  về bước đầu sau khi episode kết thúc. `γ` có tác dụng thật: nó quyết định một luật
  được "thưởng" bao nhiêu cho hậu quả muộn của nó.
- **Chính sách**: ε-greedy, ε giảm tuyến tính theo episode. `Q` khởi tạo bằng 0 trong khi
  reward ≤ 0, nên ban đầu lạc quan và tự khám phá các action chưa thử.

## 3. Công bằng ngân sách

- Môi trường xây dựng (`FJSPConstructionEnv`) mô phỏng lịch đúng ngữ nghĩa decoder
  (release date, chèn vào khe trống sớm nhất). Sau mỗi episode, nghiệm được đánh giá
  bằng `FJSPProblem.evaluate` (đếm NFE), và code **báo lỗi** nếu mục tiêu mô phỏng khác decoder.
- Bảng Q **chỉ được cập nhật sau khi nghiệm đã được đánh giá**. Không có episode
  "huấn luyện" nào chạy ngoài ngân sách.
- Nếu chromosome trùng một nghiệm đã có, nó được xây lại với ε cao hơn (tối đa
  `max_resample` lần). Bản trùng không được đánh giá, không tốn NFE và không được học.
- Tổng số nghiệm khởi tạo luôn bằng kích thước quần thể (13), nên checkpoint và lịch NFE giống baseline.

## 4. Gắn nghiệm với subproblem

Nghiệm thứ `i` được xây cho trọng số thứ `i`. `AbstractMoead` khởi tạo quần thể *trước*
khi xáo trộn `weights`, nên `MoeadQInit` hoán vị quần thể theo cùng phép xáo trộn,
và mỗi nghiệm nằm đúng subproblem nó được xây cho. Phép xáo trộn và số lần gọi RNG
không đổi.

Với `hybrid_q_random`, các subproblem dành cho Q-learning được rải đều trên lưới
trọng số và gồm cả hai đầu mút khi có từ 2 nghiệm Q-learning trở lên (mặc định
`random_ratio = 0.2` cho 3 ngẫu nhiên + 10 Q-learning). Nếu chỉ còn 1 nghiệm Q-learning
(`random_ratio` rất cao, ví dụ 0.95), nghiệm đó nằm ở subproblem giữa. Nghiệm ngẫu nhiên
lấp vào các subproblem còn lại.

## 5. Tham số (`q_init` trong config, cờ `--q-*` trong `run_fjsp.py`)

| Tham số | Mặc định | Ý nghĩa |
| --- | ---: | --- |
| `alpha` | 0.1 | tốc độ học |
| `gamma` | 0.95 | hệ số chiết khấu |
| `epsilon_start` / `epsilon_end` | 0.5 / 0.05 | ε của episode đầu / cuối |
| `random_ratio` | 0.2 | tỉ lệ nghiệm ngẫu nhiên (chỉ `hybrid_q_random`) |
| `progress_bins` / `weight_bins` | 4 / 3 | độ phân giải state |
| `max_resample` | 5 | số lần xây lại khi trùng |

Tham số nào không khai báo sẽ lấy giá trị mặc định. Giá trị sai (ví dụ `gamma = 1.5`)
bị từ chối trước khi tốn NFE nào.

## 6. Cách chạy

Chạy từ thư mục gốc, với môi trường đã cài theo [README](../README.md#1-cài-đặt)
(`pip install -e ./vendor`). Nếu dùng Python khác mà chưa cài, lệnh sẽ báo
`ModuleNotFoundError: moead_framework`.

### 6.1 Một lượt chạy

```powershell
# Hybrid Random + Q-learning, tham số mặc định
./.venv/Scripts/python.exe scripts/run_fjsp.py --initializer hybrid_q_random --seed 1

# Chỉ Q-learning, đổi tham số, lưu archive/gene/lịch ra file mới
./.venv/Scripts/python.exe scripts/run_fjsp.py --instance "instance/main/Brandimate1993/Tight/Mk10/Mk10.fjs" `
    --initializer qlearning --q-alpha 0.2 --q-gamma 0.9 --seed 3 --output results/mk10_qlearning_seed3.json
```

Không truyền `--initializer` (hoặc truyền `baseline`) thì chạy `Moead` gốc như trước.
Ngoài các trường của baseline, output in ra có thêm:

| Trường | Nội dung |
| --- | --- |
| `algorithm` | `MoeadQInit` |
| `initializer`, `q_init` | Chế độ và bộ tham số thực tế đã dùng |
| `initializer_stats.n_random` / `n_qlearning` | Số nghiệm ngẫu nhiên / do Q-learning xây (tổng = 13) |
| `initializer_stats.td_updates` | Số lần cập nhật TD (= số episode × số thao tác) |
| `initializer_stats.action_counts` | Số lần dùng từng luật ghép |
| `initializer_stats.episode_returns`, `epsilon_schedule` | Return và ε của từng episode |
| `initializer_stats.greedy_policy` | Luật greedy đã học cho từng state `w{bin λ}-p{tiến độ}-t{áp lực trễ}` |

Một lượt hợp lệ phải có `actual_evaluations` đúng bằng `--evaluations` và `all_schedules_valid: true`.

### 6.2 Chạy cả study (84 instance × 10 seed)

```powershell
./.venv/Scripts/python.exe scripts/run_baseline_validation.py --config scripts/configs/qinit_validation.json --output-dir results/qinit_main
```

Cách resume, cấu trúc thư mục output và điều kiện hợp lệ giống hệt baseline, xem
[EXPERIMENTS.md](EXPERIMENTS.md). Một study đầy đủ mất vài giờ; nếu bị gián đoạn,
chạy lại cùng lệnh với `--resume`.

Để chạy chế độ khác, sao chép `qinit_validation.json` thành file mới rồi sửa trong
`"algorithm"` (đổi cả `"study"` để manifest phân biệt được). Ví dụ, Q-learning thuần:

```json
"initializer": "qlearning",
"q_init": {"gamma": 0.9}
```

`q_init` chỉ cần khai báo những tham số muốn đổi.

### 6.3 So sánh với baseline

Hai study phải dùng cùng instance, seed và ngân sách:

```powershell
./.venv/Scripts/python.exe scripts/run_baseline_validation.py --output-dir results/baseline_main
./.venv/Scripts/python.exe scripts/compare_initializers.py --baseline baseline=results/baseline_main `
    --study hybrid=results/qinit_main --output results/compare_qinit
```

Có thể lặp `--study TÊN=THƯ_MỤC` để so sánh nhiều biến thể cùng lúc. Output gồm:

- `comparison.md`: bảng thắng/hòa/thua so với baseline (Mann–Whitney U chính xác, α = 0.05), cùng
  HV, IGD+ (reference front gộp từ mọi study), C-metric, Cmax và TWT theo từng instance;
- `comparison.csv` / `comparison.json`: cùng số liệu, kèm p-value;
- `convergence.png`: HV trung bình theo NFE của từng study.

### 6.4 Kiểm tra

```powershell
./.venv/Scripts/python.exe -m pytest vendor/moead_framework/test/test_qinit.py -q
```

Các test kiểm tra: bộ mô phỏng khớp decoder, công thức TD, `MoeadQInit` với khởi tạo
ngẫu nhiên trùng từng bit với `Moead` gốc (khung mới không đổi thuật toán), đếm NFE và
archive, tái lập theo seed, chạy end-to-end study và CLI.

## 7. Kết quả kiểm thử

Protocol giống baseline: H = 12 (13 nghiệm), T = 10, pc = 0.8, pm = 0.05, 1.000 NFE,
seed 0–9, tham số Q mặc định. Sáu instance trải đủ 2 bộ data × 3 mức hạn chót:
`Mk02` Loose, `Mk06` Moderate, `Mk09` Tight (Brandimate1993); `03a` Loose, `08a` Moderate,
`12a` Tight (DauzerePeresPaulli1994). 180/180 lượt hợp lệ, lượt lặp seed cho kết quả giống hệt.

Mỗi cột ứng với một config trong `scripts/configs/`: `qinit_pilot6_baseline.json`,
`qinit_pilot6_qlearning.json`, `qinit_pilot6_hybrid_q_random.json`. Chạy lại bảng:

```powershell
foreach ($m in "baseline", "qlearning", "hybrid_q_random") {
    ./.venv/Scripts/python.exe scripts/run_baseline_validation.py --config "scripts/configs/qinit_pilot6_$m.json" --output-dir "results/pilot6_$m"
}
./.venv/Scripts/python.exe scripts/compare_initializers.py --baseline baseline=results/pilot6_baseline `
    --study qlearning=results/pilot6_qlearning --study hybrid=results/pilot6_hybrid_q_random --output results/compare_pilot6
```

| Instance | baseline | qlearning | hybrid_q_random | Tốt nhất |
| --- | ---: | ---: | ---: | --- |
| mk02_loose | 0.8934 ± 0.0116 | 0.9691 ± 0.0053 | **0.9696** ± 0.0056 | `hybrid_q_random` |
| mk06_moderate | 0.8600 ± 0.0069 | **0.9457** ± 0.0036 | 0.9415 ± 0.0038 | `qlearning` |
| mk09_tight | 0.8855 ± 0.0092 | **0.9290** ± 0.0029 | 0.9282 ± 0.0025 | `qlearning` |
| 03a_loose | 0.8512 ± 0.0105 | **0.9294** ± 0.0051 | 0.9254 ± 0.0066 | `qlearning` |
| 08a_moderate | 0.8706 ± 0.0063 | **0.9126** ± 0.0033 | 0.9120 ± 0.0029 | `qlearning` |
| 12a_tight | 0.8468 ± 0.0050 | **0.9011** ± 0.0028 | 0.9002 ± 0.0033 | `qlearning` |

HV cuối (1.000 NFE), trung bình ± độ lệch chuẩn qua 10 seed; càng lớn càng tốt. Giá trị
in đậm là cao nhất trên dòng.

- **So với baseline:** cả `qlearning` và `hybrid_q_random` có HV cao hơn ở 6/6 instance,
  khác biệt có ý nghĩa (Mann–Whitney U, p < 0.0001). Thời gian chạy tương đương baseline.
- **`qlearning` so với `hybrid_q_random`:** `qlearning` có HV trung bình cao nhất ở 5/6
  instance, nhưng chênh lệch chỉ có ý nghĩa ở `mk06` (p ≈ 0.04); ở 5 instance còn lại hai
  chế độ xem như ngang nhau.
- **Pha tiến hóa:** sau khởi tạo Q-learning, HV chỉ tăng thêm rất ít đến hết 1.000 NFE.
  Phần lớn mức cải thiện có ngay từ 13 nghiệm đầu.
- **Giới hạn:** mới có 6/84 instance. Cần chạy study đầy đủ (mục 6.2, 6.3) trước khi kết luận chung.
