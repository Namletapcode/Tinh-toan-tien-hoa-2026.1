# FJSP: mô hình, kiến trúc và tích hợp module

[README](../README.md) · [Thực nghiệm](EXPERIMENTS.md) · [Dữ liệu](BENCHMARKS.md)

## 1. Mô hình và mục tiêu

FJSP tĩnh: job và máy sẵn sàng tại thời điểm 0, công đoạn theo thứ tự trong job,
mỗi máy xử lý tối đa một công đoạn tại một thời điểm, công đoạn không bị ngắt.
Mỗi công đoạn chọn một máy trong danh sách hợp lệ. Chưa xét setup, vận chuyển,
deadline, sự cố hoặc thời điểm đến khác 0. Thời gian xử lý là số nguyên không âm.

| Mục tiêu, theo thứ tự trong `solution.F` | Cách tính |
| --- | --- |
| `makespan` | Thời điểm kết thúc lớn nhất |
| `total_workload` | Tổng thời gian gia công, không gồm thời gian nhàn rỗi |
| `max_machine_workload` | Tổng thời gian gia công lớn nhất trên một máy |

Cả ba cùng minimize. Giữ nguyên MS và đổi OS có thể đổi makespan; hai mục tiêu
tải chỉ phụ thuộc cách gán máy. Mô tả dữ liệu trong
[BENCHMARKS](BENCHMARKS.md).

## 2. Biểu diễn OS/MS (mã hóa) và giải mã

Với `O` công đoạn, vector quyết định là `OS + MS`, dài `2 * O`, dùng gene nguyên:

- **OS:** mỗi gene là mã job 0-based. Job `j` xuất hiện đúng số công đoạn của nó;
  lần xuất hiện thứ `k` chọn công đoạn `(j,k)`.
- **MS:** chỉ số lựa chọn máy 0-based, xếp cố định theo job/công đoạn:
  `(job 0, op 0), ... (job 1, op 0), ...`; không xếp theo vị trí trong OS.
- MS là vị trí trong danh sách máy hợp lệ. Nếu danh sách là `(M5,4), (M2,7)`,
  MS=0 chọn M5; MS=1 chọn M2. Mã máy vật lý trong input được đổi từ 1-based
  thành 0-based khi đọc dữ liệu.

`Chromosome.from_vector(instance, vector)` kiểm tra gene và tách hai phần;
`to_vector()` ghép lại. `random_chromosome(instance, rng)` tạo OS/MS khả thi
nhưng chưa giải mã. Instance và chromosome là cấu trúc bất biến.

**Giải mã**:
`decode(instance, chromosome)` duyệt OS, tra MS theo vị trí cố định của công
đoạn và chèn vào khoảng trống sớm nhất sau công đoạn trước cùng job. Có xét
khoảng trống đầu máy, giữa các công đoạn và sau công đoạn cuối. OS là thứ tự
ưu tiên đưa công đoạn vào lịch; công đoạn xét sau có thể bắt đầu sớm hơn trên
máy khác. Các công đoạn đã xếp không bị dịch chuyển.

Cho phép tiếp giáp `end_trước <= start_sau`. Công đoạn thời gian 0 có thể nằm
ở đầu/cuối một khoảng bận nhưng không nằm bên trong khoảng máy đang bận.

Kết quả `Schedule` giữ máy, start/end, tải máy và `objectives`.
`validate_schedule()` kiểm tra độc lập: đủ công đoạn, đúng máy/thời gian,
thứ tự trong job, không chồng lấn máy và tải khớp lịch.

## 3. Kiến trúc baseline

`create_fjsp_moead()` lắp các component FJSP vào lớp `Moead` cơ bản.

| File trong `vendor/moead_framework/` | Vai trò |
| --- | --- |
| [algorithm/abstract_moead.py](../vendor/moead_framework/algorithm/abstract_moead.py) | Vòng lặp, ngân sách, láng giềng, điểm lý tưởng, quần thể và archive |
| [algorithm/moead.py](../vendor/moead_framework/algorithm/moead.py) | Lớp MOEA/D dùng để lắp các component |
| [algorithm/fjsp.py](../vendor/moead_framework/algorithm/fjsp.py) | Factory FJSP và trọng số simplex ba chiều |
| [problem/fjsp/instance.py](../vendor/moead_framework/problem/fjsp/instance.py) | Bộ đọc và cấu trúc instance |
| [problem/fjsp/encoding.py](../vendor/moead_framework/problem/fjsp/encoding.py) | Chromosome và kiểm tra OS/MS |
| [problem/fjsp/decoder.py](../vendor/moead_framework/problem/fjsp/decoder.py) | Decoder, lịch và kiểm tra khả thi |
| [problem/fjsp/problem.py](../vendor/moead_framework/problem/fjsp/problem.py) | `FJSPProblem`, tạo nghiệm và đánh giá |
| [core/genetic_operator/fjsp.py](../vendor/moead_framework/core/genetic_operator/fjsp.py) | `FJSPOperatorConfig` và toán tử OS/MS |
| [core/offspring_generator/fjsp.py](../vendor/moead_framework/core/offspring_generator/fjsp.py) | Chọn cha mẹ, gọi toán tử và đánh giá con |

Một lượt tối ưu: chọn bài toán con → chọn tập ứng viên cha mẹ → sinh/đánh giá
con → cập nhật `z` → thay nghiệm quần thể → cập nhật archive. Bộ sinh con gọi
bộ chọn cha mẹ, sao chép gene, gọi toán tử rồi `problem.evaluate()`.

| Component | API chính | Baseline chọn |
| --- | --- | --- |
| `sps_strategy` | `get_sub_problems()` trả chỉ số bài toán con | `SpsAllSubproblems` |
| `mating_pool_selector` | `select(sub_problem)` trả danh sách chỉ số ứng viên | `ClosestNeighborsSelector` |
| `parent_selector` | `select(indexes)` trả các đối tượng nghiệm cha mẹ | `TwoRandomParentSelector` |
| `genetic_operator` | `run()` trả vector gene con | `FJSPCrossoverAndMutation` |
| `offspring_generator` | `run(population_indexes)` trả nghiệm con đã đánh giá | `FJSPOffspringGenerator` |
| `termination_criteria` | `test()` trả true khi còn được chạy | `MaxEvaluation` |

Láng giềng dựa trên khoảng cách Euclid giữa vector trọng số. Hai lần lấy cha
mẹ ngẫu nhiên có thể chọn trùng nghiệm.

Toán tử baseline: POX cho OS, lai đều từng gene MS, đột biến hoán đổi hai vị trí
OS và đổi một MS sang lựa chọn máy khác hợp lệ. `pc` áp dụng cho bước lai OS/MS;
hai phần OS và MS có lần thử đột biến độc lập với cùng `pm` trên mỗi con.
Khởi tạo ngẫu nhiên; các xác suất giữ cố định.

## 4. Hợp đồng luồng hoạt động phải giữ khi tích hợp idea

- `N` là số trọng số và kích thước quần thể; `1 <= T <= N`. Ngân sách là số
  nguyên `>= N`. API trọng số yêu cầu hữu hạn, không âm, đúng số mục tiêu và
  tổng từng vector dương; factory FJSP sinh lưới simplex tổng bằng 1.
  Thuật toán dừng ngay khi hết ngân sách, kể cả giữa lượt duyệt bài toán con.
- `z` bắt đầu bằng cực tiểu từng mục tiêu của quần thể đầu và cập nhật từ các
  nghiệm quan sát được. Baseline dùng Tchebycheff với mục tiêu gốc, không
  chuẩn hóa trong thuật toán.
- **Một NFE** là một lần đánh giá đủ ba mục tiêu của một ứng viên, gồm cả khởi
  tạo và nghiệm lặp. `FJSPProblem.evaluate()` giải mã một lần, tăng
  `evaluation_count` một và lưu gene, `F`, `schedule`.
- Dùng `evaluate()` khi tạo con. Gọi riêng `f(i, vector)` cũng tiêu một lần
  đánh giá đầy đủ; vòng gọi `f()` cho từng mục tiêu sẽ tiêu nhiều NFE.
- Factory yêu cầu một `FJSPProblem` mới cho mỗi lượt. `validate_solution()`
  kiểm tra gene, mục tiêu và lịch đã lưu mà không giải mã lại/tăng NFE.
- Archive `ep` giữ các vector không trội trong **mọi nghiệm đã đánh giá**,
  kể cả con không được nhận vào quần thể. Bỏ trùng theo vector mục tiêu và
  giữ một gene/lịch đại diện; archive không dùng để chọn cha mẹ của baseline.
- Gene cha mẹ được sao chép trước biến đổi; nghiệm thay vào quần thể và lưu
  archive được sao chép độc lập. Module không được sửa trực tiếp cha mẹ.
- Đổi gene làm mất hiệu lực của mục tiêu/lịch cũ. Cần đánh giá lại và tính NFE.
  `repair()` trong vòng lặp nhận nghiệm đã đánh giá, mặc định trả nguyên nghiệm.
- Vòng lặp hiện tăng một NFE cho mỗi lần sinh con. Nếu module có local search,
  nhiều ứng viên hoặc repair có đánh giá, phải sửa phần điều phối/bộ đếm để
  tính **mọi** đánh giá và dừng đúng ngân sách; thay bộ sinh con đơn thuần chưa
  đủ. Xem các ca hồi quy về ngân sách trước khi triển khai.

## 5. Tích hợp và triển khai module mới.
**Lưu ý**: Đặt module mới ở file riêng và lắp trong entrypoint thực nghiệm,
giữ đường chạy baseline để so sánh

| Ý tưởng | Điểm mở rộng |
| --- | --- |
| Lai/đột biến, bảo toàn OS/MS | `core/genetic_operator/`; nhận `solutions, instance, config` và trả vector nguyên khả thi |
| Chọn cha mẹ hoặc tập giao phối | `core/parent_selector/` hoặc `core/selector/` |
| Chọn hướng trọng số cần tối ưu | `core/sps_strategy/` |
| Tổ chức sinh con, chọn giữa nhiều ứng viên | `core/offspring_generator/` và điều phối NFE khi có nhiều đánh giá |
| Khởi tạo nghiệm | Kế thừa `FJSPProblem` và thay `generate_random_solution()`, vẫn đánh giá/đếm từng nghiệm |
| Phân rã, điều kiện dừng hoặc vòng lặp | `aggregation/`, `core/termination_criteria/` hoặc lớp thuật toán riêng |
| Định nghĩa mục tiêu/decoder | `problem/fjsp/`; ghi rõ đây là thay đổi mô hình khi đối chiếu kết quả |

Ví dụ nối một lớp toán tử vào pipeline, chạy từ gốc project:

```python
from moead_framework.algorithm.fjsp import create_fjsp_moead
from moead_framework.core.genetic_operator.fjsp import FJSPCrossoverAndMutation
from moead_framework.problem.fjsp import FJSPProblem
from moead_framework.tool.result import set_seed

class CustomOperator(FJSPCrossoverAndMutation):
    """Kế thừa constructor; override run() khi triển khai ý tưởng."""

set_seed(1)
problem = FJSPProblem("instance/FattahiMehrabadJolai2007/fattahi1.txt")
algorithm = create_fjsp_moead(
    problem, partitions=2, number_of_weight_neighborhood=3, max_evaluation=13,
)
algorithm.genetic_operator = CustomOperator  # Trường này nhận lớp toán tử.
archive = algorithm.run()
```

Ví dụ trên kế thừa hành vi baseline. Khi thay component khác, các trường như
`parent_selector` và `offspring_generator` nhận đối tượng gắn với algorithm;
xem constructor của component tương ứng. Factory chưa nhận mọi component làm
tham số trực tiếp.

Sau khi tích hợp các module mới, cần test để kiểm tra tính khả thi/hợp lý của cả module hoặc cả luồng. Test component mới trong `vendor/moead_framework/test/`: gene khả thi, không sửa
cha mẹ, mục tiêu/lịch khớp, đúng ngân sách và tái lập seed. Các bộ kiểm tra chung:

| File test | Phạm vi |
| --- | --- |
| `test_fjsp.py` | Input, OS/MS, decoder, lịch và ví dụ nhỏ có oracle độc lập |
| `test_fjsp_integration.py` | Toán tử, đánh giá một lần, quần thể/archive và CLI |
| `test_core_regressions.py` | NFE, `z`, sao chép nghiệm, archive và láng giềng |
| `test_fjsp_experiments.py` | HV, ghi kết quả, tiếp tục chạy và kiểm tra checksum |

## 6. Ví dụ giải mã tính tay

```powershell
./.venv/Scripts/python.exe scripts/smoke_fjsp.py --os "0,1,0,1" --ms "1,1,0,0" --show-operations
```

Với `fattahi1`, nhãn trong bảng là 1-based để đọc:

| Công đoạn | Máy | Start | End |
| --- | --- | ---: | ---: |
| J1/O1 | M2 | 0 | 37 |
| J2/O1 | M1 | 0 | 45 |
| J1/O2 | M2 | 37 | 61 |
| J2/O2 | M1 | 45 | 66 |

Tải máy `(66,61)`, ba mục tiêu `(66,127,66)`. Đây là giải mã chromosome đã cho.
JSON và API dùng chỉ số 0-based. Giới hạn thực nghiệm và cách đánh giá module
xem [EXPERIMENTS.md](EXPERIMENTS.md).
