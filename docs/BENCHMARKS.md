# Dữ liệu FJSP và mốc benchmark makespan

[README](../README.md) · [Mô hình FJSP](FJSP.md) · [Thực nghiệm](EXPERIMENTS.md)

Tài liệu tra cứu định dạng input, các họ instance và bảng LB/BKS.
Mô hình ba mục tiêu, OS/MS và decoder nằm ở [FJSP](FJSP.md).

**LB/BKS ở đây chỉ tham chiếu makespan, không phải front chuẩn cho ba mục tiêu.**
Nguồn là [FJSPLIB](https://scheduleopt.github.io/benchmarks/fjsplib#best-known-solutions-per-instance-family);
[mô hình tham chiếu được chốt](https://github.com/ScheduleOpt/benchmarks/blob/8a6e36ef31263d36975e56fb481e0ba04e188d6f/flexible-jobshop/code/optalcp/flexiblejobshop.mjs#L42-L47)
tối thiểu hóa thời điểm hoàn thành tất cả job. Các bảng dưới đây giữ snapshot
nguồn và phạm vi instance nêu trong tài liệu, không tự cập nhật theo nguồn trực tuyến.

## 1. Cấu trúc input

Mỗi file `.txt` trong thư mục [`instance/`](../instance/) mô tả một bài toán độc lập và có dạng:

```text
n_jobs n_machines metadata
n_ops_job1 k_11 machine_id processing_time ... k_12 machine_id processing_time ...
n_ops_job2 k_21 machine_id processing_time ... k_22 machine_id processing_time ...
...
```

`metadata` là tùy chọn đối với bộ đọc của project; nếu có, giá trị này chỉ dùng
để tham khảo, không quyết định danh sách máy hợp lệ của công đoạn.

| Trường | Ý nghĩa |
| --- | --- |
| `n_jobs` | Số job |
| `n_machines` | Tổng số máy |
| `metadata` | Tham số mô tả độ linh hoạt, dùng để tham khảo |
| `n_ops_jobi` | Số công đoạn của job `i` |
| `k_ij` | Số máy có thể thực hiện công đoạn `j` của job `i` |
| `machine_id processing_time` | Mã máy và thời gian xử lý trên máy đó |

Sau dòng đầu, **mỗi dòng tương ứng với một job**. Sau số công đoạn, lần lượt đọc từng công đoạn: đọc `k_ij`, rồi đọc đúng `k_ij` cặp `(machine_id, processing_time)`. Thứ tự các công đoạn trên dòng chính là thứ tự bắt buộc trong job. Các máy trong một công đoạn là những phương án thay thế; lời giải chỉ chọn một máy.

### Ví dụ cụ thể

File [`fattahi1.txt`](../instance/FattahiMehrabadJolai2007/fattahi1.txt):

```text
2 2 2
2 2 1 25 2 37 2 1 32 2 24
2 2 1 45 2 65 2 1 21 2 65
```

Instance này có 2 job, 2 máy và tổng cộng 4 công đoạn. Dòng job thứ nhất được tách thành:

```text
2 | 2 (1,25) (2,37) | 2 (1,32) (2,24)
    Công đoạn O11     Công đoạn O12
```

| Job | Công đoạn | Thời gian trên M1 | Thời gian trên M2 |
| --- | --- | ---: | ---: |
| J1 | O11 | 25 | 37 |
| J1 | O12 | 32 | 24 |
| J2 | O21 | 45 | 65 |
| J2 | O22 | 21 | 65 |

Ví dụ, `O11` có thể chạy trên M1 trong 25 đơn vị thời gian hoặc M2 trong 37 đơn vị thời gian. Các quan hệ thứ tự là `O11 → O12` và `O21 → O22`. Thời điểm bắt đầu và kết thúc chưa có trong input; chúng được xác định khi lập lịch.

### Lưu ý khi đọc dữ liệu

- Mã máy trong file bắt đầu từ **1**. Các giá trị thời gian được hiểu theo cùng một đơn vị thời gian quy ước; file không ghi đơn vị vật lý.
- Số công đoạn có thể khác nhau giữa các job; số máy phù hợp có thể khác nhau giữa các công đoạn.
- Dùng khoảng trắng bất kỳ để tách token vì dữ liệu chứa cả dấu cách và tab; bỏ qua dòng trống.
- `metadata` có thể là số thực và không luôn bằng số máy thay thế trung bình tính từ dữ liệu. Ví dụ, `mk01.txt` ghi `10 6 2` nhưng có 115 lựa chọn máy cho 55 công đoạn, trung bình khoảng `2,09`. Bộ đọc phải dựa vào `k_ij` của từng công đoạn.
- File `orb7.txt` trong cả bốn nhóm Hurink có công đoạn với thời gian xử lý bằng **0**. Cần giữ nguyên dữ liệu và xử lý trường hợp này khi giải mã lịch.

## 2. Các bộ instance sử dụng

Thư mục hiện có **400 instance thuộc 7 bộ dữ liệu nguồn**; riêng Hurink được chia thành 4 nhóm. Các khoảng giá trị dưới đây được thống kê trực tiếp từ những file đang có trong repository. Số công đoạn là tổng số công đoạn của một instance.

| Bộ dữ liệu / thư mục | Số file | Số job | Số máy | Số công đoạn | Đặc điểm |
| --- | ---: | ---: | ---: | ---: | --- |
| [KacemHammadiBorne2002](../instance/KacemHammadiBorne2002/) | 4 | 4–15 | 5–10 | 12–56 | Mọi công đoạn đều có thể chạy trên mọi máy; thời gian phụ thuộc máy. Phù hợp minh họa việc gán máy và kiểm tra lời giải. |
| [FattahiMehrabadJolai2007](../instance/FattahiMehrabadJolai2007/) | 20 | 2–12 | 2–8 | 4–48 | Quy mô nhỏ, tập máy lựa chọn thay đổi theo công đoạn; phù hợp kiểm tra bộ đọc và thuật toán trên ví dụ dễ theo dõi. |
| [Brandimarte1993](../instance/Brandimarte1993/) | 15 | 10–30 | 4–15 | 55–284 | Gồm `mk01`–`mk15`, độ linh hoạt khác nhau và thời gian phụ thuộc máy; dùng đánh giá đồng thời quyết định gán máy và thứ tự xử lý. |
| [DauzerePeresPaulli1994](../instance/DauzerePeresPaulli1994/) | 18 | 10–20 | 5–10 | 196–387 | Gồm `01a`–`18a`, nhiều công đoạn trong mỗi instance; số lựa chọn máy và thời gian xử lý thay đổi. |
| [ChambersBarnes1996](../instance/ChambersBarnes1996/) | 19 | 10–15 | 11–18 | 100–225 | Độ linh hoạt thấp, trung bình khoảng 1,07–1,30 máy/công đoạn; một công đoạn có cùng thời gian trên các máy phù hợp. |
| [BehnkeGeiger2012](../instance/BehnkeGeiger2012/) | 60 | 10–100 | 20–60 | 50–500 | Nhiều máy và nhiều phương án gán máy, trung bình khoảng 6,08–22,24 máy/công đoạn; phù hợp đánh giá khả năng mở rộng. |
| [Hurink — sdata](../instance/HurinkJurischThole1994/sdata/) | 66 | 6–30 | 4–15 | 36–300 | Mỗi công đoạn chỉ có một máy phù hợp; tương ứng trường hợp Job Shop không có lựa chọn máy. |
| [Hurink — edata](../instance/HurinkJurischThole1994/edata/) | 66 | 6–30 | 4–15 | 36–300 | Độ linh hoạt thấp, trung bình khoảng 1,13–1,21 máy/công đoạn. |
| [Hurink — rdata](../instance/HurinkJurischThole1994/rdata/) | 66 | 6–30 | 4–15 | 36–300 | Độ linh hoạt cao hơn `edata`, trung bình khoảng 1,88–2,08 máy/công đoạn. |
| [Hurink — vdata](../instance/HurinkJurischThole1994/vdata/) | 66 | 6–30 | 4–15 | 36–300 | Nhiều lựa chọn máy hơn, trung bình khoảng 2,09–6,70 máy/công đoạn. |

Với mỗi nhóm, khoảng số máy/công đoạn được tính bằng cách lấy trung bình trong từng instance, rồi lấy giá trị nhỏ nhất và lớn nhất của nhóm; không lấy trực tiếp từ `metadata`.

Các bộ khác nhau về quy mô và khả năng lựa chọn máy, nhưng đều được đánh giá theo cùng mục tiêu **tối thiểu hóa makespan** trong phạm vi benchmark đang sử dụng. Với Hurink và Chambers–Barnes, dù thời gian xử lý không phụ thuộc máy, việc lựa chọn máy và thứ tự xử lý vẫn có thể tác động đến makespan. Riêng nhóm `sdata` đã cố định máy cho từng công đoạn, nên thuật toán chỉ cần tối ưu thứ tự xử lý.

## 3. Best known solution và cách so sánh

Nguồn tham chiếu: [FJSPLIB — Best known solutions per instance family](https://scheduleopt.github.io/benchmarks/fjsplib#best-known-solutions-per-instance-family). Các giá trị dưới đây được chốt ngày **10/11/2026**, từ [file `bks.json` tại commit `8a6e36e`](https://github.com/ScheduleOpt/benchmarks/blob/8a6e36ef31263d36975e56fb481e0ba04e188d6f/flexible-jobshop/solutions/bks.json) để có thể tái lập việc so sánh khi nguồn cập nhật.

### 3.1. Ý nghĩa các mốc tham chiếu

Các mốc này áp dụng cho bài toán **tối thiểu hóa makespan `C_max`**:

- **LB (lower bound):** cận dưới tốt nhất mà nguồn ghi nhận.
- **UB (upper bound):** makespan của lời giải khả thi tốt nhất mà nguồn ghi nhận; dùng làm **BKS** để so sánh.
- **LB = UB:** nguồn ghi nhận giá trị tối ưu. Đây là thông tin từ nguồn, chưa phải chứng minh được kiểm tra lại trong dự án.
- **LB < UB:** vẫn còn khoảng cách giữa hai cận; đạt BKS chưa có nghĩa là đạt tối ưu.

Bảng tổng hợp chỉ tính những file đang có trong dự án:

| Bộ / nhóm instance | Số file có BKS | LB = UB | LB < UB |
| --- | ---: | ---: | ---: |
| Kacem | 4 | 4 | 0 |
| Fattahi | 20 | 20 | 0 |
| Brandimarte | 15 | 14 | 1 |
| Dauzère-Pérès–Paulli | 18 | 8 | 10 |
| Chambers–Barnes | 19 | 19 | 0 |
| Behnke–Geiger | 60 | 33 | 27 |
| Hurink / sdata | 66 | 66 | 0 |
| Hurink / edata | 66 | 64 | 2 |
| Hurink / rdata | 66 | 54 | 12 |
| Hurink / vdata | 66 | 65 | 1 |

**Đối chiếu tên và dữ liệu:** cả 400 file cục bộ khớp nội dung với bản FJSPLIB tại commit trên sau khi chuẩn hóa xuống dòng. Với Dauzère-Pérès–Paulli, file `01a.txt` tương ứng khóa `dpp01a` trong JSON; với Hurink, `rdata/la01.txt` tương ứng `la01_rdata`. Chambers–Barnes trong dự án có 19/21 instance của nguồn, thiếu `seti5xy` và `seti5xyz`.

Nguồn ghi kích thước `kacem1` là `4 × 6` và `fattahi6` là `3 × 2`, trong khi header file thực tế lần lượt là **`4 × 5`** và **`3 × 3`**. Do nội dung file khớp bản nguồn, vẫn sử dụng BKS theo đúng tên instance; kích thước input lấy từ file.

### 3.2. Giá trị LB và BKS cho từng instance

Mở từng bảng để tra cứu. Các số là đơn vị thời gian theo input; BKS thuộc về từng instance riêng lẻ, không có một giá trị BKS chung cho cả bộ.

<details>
<summary>Kacem — 4 instance</summary>

| Instance trong dự án | LB | BKS (`UB`) |
| --- | ---: | ---: |
| `kacem1.txt` | 11 | 11 |
| `kacem2.txt` | 11 | 11 |
| `kacem3.txt` | 7 | 7 |
| `kacem4.txt` | 11 | 11 |

</details>

<details>
<summary>Fattahi — 20 instance</summary>

| Instance trong dự án | LB | BKS (`UB`) |
| --- | ---: | ---: |
| `fattahi1.txt` | 66 | 66 |
| `fattahi2.txt` | 107 | 107 |
| `fattahi3.txt` | 221 | 221 |
| `fattahi4.txt` | 355 | 355 |
| `fattahi5.txt` | 119 | 119 |
| `fattahi6.txt` | 320 | 320 |
| `fattahi7.txt` | 397 | 397 |
| `fattahi8.txt` | 253 | 253 |
| `fattahi9.txt` | 210 | 210 |
| `fattahi10.txt` | 516 | 516 |
| `fattahi11.txt` | 468 | 468 |
| `fattahi12.txt` | 446 | 446 |
| `fattahi13.txt` | 466 | 466 |
| `fattahi14.txt` | 554 | 554 |
| `fattahi15.txt` | 514 | 514 |
| `fattahi16.txt` | 634 | 634 |
| `fattahi17.txt` | 879 | 879 |
| `fattahi18.txt` | 884 | 884 |
| `fattahi19.txt` | 1055 | 1055 |
| `fattahi20.txt` | 1196 | 1196 |

</details>

<details>
<summary>Brandimarte — 15 instance</summary>

| Instance trong dự án | LB | BKS (`UB`) |
| --- | ---: | ---: |
| `mk01.txt` | 40 | 40 |
| `mk02.txt` | 26 | 26 |
| `mk03.txt` | 204 | 204 |
| `mk04.txt` | 60 | 60 |
| `mk05.txt` | 172 | 172 |
| `mk06.txt` | 57 | 57 |
| `mk07.txt` | 139 | 139 |
| `mk08.txt` | 523 | 523 |
| `mk09.txt` | 307 | 307 |
| `mk10.txt` | 189 | 193 |
| `mk11.txt` | 609 | 609 |
| `mk12.txt` | 508 | 508 |
| `mk13.txt` | 383 | 383 |
| `mk14.txt` | 694 | 694 |
| `mk15.txt` | 333 | 333 |

</details>

<details>
<summary>Dauzère-Pérès–Paulli — 18 instance</summary>

| Instance trong dự án | LB | BKS (`UB`) |
| --- | ---: | ---: |
| `01a.txt` | 2505 | 2505 |
| `02a.txt` | 2228 | 2228 |
| `03a.txt` | 2228 | 2228 |
| `04a.txt` | 2503 | 2503 |
| `05a.txt` | 2195 | 2199 |
| `06a.txt` | 2164 | 2169 |
| `07a.txt` | 2216 | 2254 |
| `08a.txt` | 2061 | 2061 |
| `09a.txt` | 2061 | 2061 |
| `10a.txt` | 2212 | 2241 |
| `11a.txt` | 2019 | 2037 |
| `12a.txt` | 1969 | 1984 |
| `13a.txt` | 2206 | 2236 |
| `14a.txt` | 2161 | 2161 |
| `15a.txt` | 2161 | 2161 |
| `16a.txt` | 2202 | 2231 |
| `17a.txt` | 2089 | 2105 |
| `18a.txt` | 2057 | 2070 |

</details>

<details>
<summary>Chambers–Barnes — 19 instance</summary>

| Instance trong dự án | LB | BKS (`UB`) |
| --- | ---: | ---: |
| `mt10c1.txt` | 927 | 927 |
| `mt10cc.txt` | 908 | 908 |
| `mt10x.txt` | 918 | 918 |
| `mt10xx.txt` | 918 | 918 |
| `mt10xxx.txt` | 918 | 918 |
| `mt10xy.txt` | 905 | 905 |
| `mt10xyz.txt` | 847 | 847 |
| `setb4c9.txt` | 914 | 914 |
| `setb4cc.txt` | 907 | 907 |
| `setb4x.txt` | 925 | 925 |
| `setb4xx.txt` | 925 | 925 |
| `setb4xxx.txt` | 925 | 925 |
| `setb4xy.txt` | 910 | 910 |
| `setb4xyz.txt` | 902 | 902 |
| `seti5c12.txt` | 1169 | 1169 |
| `seti5cc.txt` | 1135 | 1135 |
| `seti5x.txt` | 1198 | 1198 |
| `seti5xx.txt` | 1194 | 1194 |
| `seti5xxx.txt` | 1194 | 1194 |

</details>

<details>
<summary>Behnke–Geiger — 60 instance</summary>

| Instance trong dự án | LB | BKS (`UB`) |
| --- | ---: | ---: |
| `behnke1.txt` | 90 | 90 |
| `behnke2.txt` | 91 | 91 |
| `behnke3.txt` | 91 | 91 |
| `behnke4.txt` | 97 | 97 |
| `behnke5.txt` | 91 | 91 |
| `behnke6.txt` | 125 | 125 |
| `behnke7.txt` | 117 | 124 |
| `behnke8.txt` | 123 | 123 |
| `behnke9.txt` | 125 | 125 |
| `behnke10.txt` | 127 | 127 |
| `behnke11.txt` | 223 | 228 |
| `behnke12.txt` | 213 | 219 |
| `behnke13.txt` | 223 | 229 |
| `behnke14.txt` | 221 | 230 |
| `behnke15.txt` | 220 | 228 |
| `behnke16.txt` | 392 | 412 |
| `behnke17.txt` | 392 | 401 |
| `behnke18.txt` | 390 | 396 |
| `behnke19.txt` | 395 | 400 |
| `behnke20.txt` | 391 | 398 |
| `behnke21.txt` | 85 | 85 |
| `behnke22.txt` | 87 | 87 |
| `behnke23.txt` | 85 | 85 |
| `behnke24.txt` | 87 | 87 |
| `behnke25.txt` | 87 | 87 |
| `behnke26.txt` | 113 | 113 |
| `behnke27.txt` | 122 | 122 |
| `behnke28.txt` | 114 | 114 |
| `behnke29.txt` | 117 | 117 |
| `behnke30.txt` | 120 | 120 |
| `behnke31.txt` | 226 | 226 |
| `behnke32.txt` | 220 | 224 |
| `behnke33.txt` | 224 | 224 |
| `behnke34.txt` | 219 | 223 |
| `behnke35.txt` | 211 | 214 |
| `behnke36.txt` | 381 | 388 |
| `behnke37.txt` | 387 | 391 |
| `behnke38.txt` | 387 | 389 |
| `behnke39.txt` | 384 | 389 |
| `behnke40.txt` | 415 | 419 |
| `behnke41.txt` | 87 | 87 |
| `behnke42.txt` | 87 | 87 |
| `behnke43.txt` | 86 | 86 |
| `behnke44.txt` | 84 | 84 |
| `behnke45.txt` | 87 | 87 |
| `behnke46.txt` | 114 | 114 |
| `behnke47.txt` | 117 | 117 |
| `behnke48.txt` | 125 | 125 |
| `behnke49.txt` | 113 | 113 |
| `behnke50.txt` | 123 | 123 |
| `behnke51.txt` | 215 | 218 |
| `behnke52.txt` | 211 | 211 |
| `behnke53.txt` | 212 | 215 |
| `behnke54.txt` | 221 | 223 |
| `behnke55.txt` | 221 | 223 |
| `behnke56.txt` | 384 | 390 |
| `behnke57.txt` | 385 | 390 |
| `behnke58.txt` | 392 | 397 |
| `behnke59.txt` | 392 | 398 |
| `behnke60.txt` | 397 | 397 |

</details>

<details>
<summary>Hurink — 66 instance gốc × 4 nhóm = 264 instance</summary>

Mỗi ô ghi `LB / UB`. Khi hai cận bằng nhau, chỉ ghi một số. Cần chọn đúng cột ứng với thư mục `sdata`, `edata`, `rdata` hoặc `vdata`.

| Tên file | sdata | edata | rdata | vdata |
| --- | ---: | ---: | ---: | ---: |
| `abz5.txt` | 1234 | 1167 | 954 | 859 |
| `abz6.txt` | 943 | 925 | 807 | 742 |
| `abz7.txt` | 656 | 604 / 610 | 497 / 522 | 492 |
| `abz8.txt` | 667 | 625 / 636 | 509 / 535 | 506 / 507 |
| `abz9.txt` | 678 | 644 | 517 / 536 | 497 |
| `car1.txt` | 7038 | 6176 | 5034 | 5005 |
| `car2.txt` | 7166 | 6327 | 5985 | 5929 |
| `car3.txt` | 7312 | 6856 | 5622 | 5597 |
| `car4.txt` | 8003 | 7789 | 6514 | 6514 |
| `car5.txt` | 7702 | 7229 | 5615 | 4909 |
| `car6.txt` | 8313 | 7990 | 6147 | 5486 |
| `car7.txt` | 6558 | 6123 | 4425 | 4281 |
| `car8.txt` | 8264 | 7689 | 5692 | 4613 |
| `ft06.txt` | 55 | 55 | 47 | 47 |
| `ft10.txt` | 930 | 871 | 686 | 655 |
| `ft20.txt` | 1165 | 1088 | 1022 | 1022 |
| `la01.txt` | 666 | 609 | 570 | 570 |
| `la02.txt` | 655 | 655 | 529 | 529 |
| `la03.txt` | 597 | 550 | 477 | 477 |
| `la04.txt` | 590 | 568 | 502 | 502 |
| `la05.txt` | 593 | 503 | 457 | 457 |
| `la06.txt` | 926 | 833 | 799 | 799 |
| `la07.txt` | 890 | 762 | 749 | 749 |
| `la08.txt` | 863 | 845 | 765 | 765 |
| `la09.txt` | 951 | 878 | 853 | 853 |
| `la10.txt` | 958 | 866 | 804 | 804 |
| `la11.txt` | 1222 | 1103 | 1071 | 1071 |
| `la12.txt` | 1039 | 960 | 936 | 936 |
| `la13.txt` | 1150 | 1053 | 1038 | 1038 |
| `la14.txt` | 1292 | 1123 | 1070 | 1070 |
| `la15.txt` | 1207 | 1111 | 1089 | 1089 |
| `la16.txt` | 945 | 892 | 717 | 717 |
| `la17.txt` | 784 | 707 | 646 | 646 |
| `la18.txt` | 848 | 842 | 666 | 663 |
| `la19.txt` | 842 | 796 | 700 | 617 |
| `la20.txt` | 902 | 857 | 756 | 756 |
| `la21.txt` | 1046 | 1009 | 809 / 825 | 800 |
| `la22.txt` | 927 | 880 | 745 / 753 | 733 |
| `la23.txt` | 1032 | 950 | 820 / 831 | 809 |
| `la24.txt` | 935 | 908 | 780 / 795 | 773 |
| `la25.txt` | 977 | 936 | 771 / 779 | 751 |
| `la26.txt` | 1218 | 1106 | 1056 / 1057 | 1052 |
| `la27.txt` | 1235 | 1181 | 1085 | 1084 |
| `la28.txt` | 1216 | 1142 | 1075 / 1076 | 1069 |
| `la29.txt` | 1152 | 1107 | 993 / 994 | 993 |
| `la30.txt` | 1355 | 1188 | 1068 / 1071 | 1068 |
| `la31.txt` | 1784 | 1532 | 1520 | 1520 |
| `la32.txt` | 1850 | 1698 | 1657 | 1657 |
| `la33.txt` | 1719 | 1547 | 1497 | 1497 |
| `la34.txt` | 1721 | 1599 | 1535 | 1535 |
| `la35.txt` | 1888 | 1736 | 1549 | 1549 |
| `la36.txt` | 1268 | 1160 | 1023 | 948 |
| `la37.txt` | 1397 | 1397 | 1062 | 986 |
| `la38.txt` | 1196 | 1141 | 954 | 943 |
| `la39.txt` | 1233 | 1184 | 1011 | 922 |
| `la40.txt` | 1222 | 1144 | 955 | 955 |
| `orb1.txt` | 1059 | 977 | 746 | 695 |
| `orb2.txt` | 888 | 865 | 696 | 620 |
| `orb3.txt` | 1005 | 951 | 712 | 648 |
| `orb4.txt` | 1005 | 984 | 753 | 753 |
| `orb5.txt` | 887 | 842 | 639 | 584 |
| `orb6.txt` | 1010 | 958 | 754 | 715 |
| `orb7.txt` | 397 | 389 | 302 | 275 |
| `orb8.txt` | 899 | 894 | 639 | 573 |
| `orb9.txt` | 934 | 933 | 694 | 659 |
| `orb10.txt` | 944 | 933 | 742 | 681 |

</details>

### 3.3. Cách báo cáo kết quả so với BKS

Với một lịch khả thi có makespan `C`, tính độ lệch tương đối:

```text
RPD_BKS (%) = 100 × (C - UB) / UB
```

- `RPD_BKS = 0`: đạt best known theo bản dữ liệu đã chốt; chỉ kết luận tối ưu khi `LB = UB`.
- `RPD_BKS > 0`: makespan lớn hơn BKS; càng nhỏ càng tốt.
- `RPD_BKS < 0`: có thể cải thiện BKS đã chốt; cần kiểm tra tính khả thi của lịch và sự trùng khớp dữ liệu trước khi báo cáo.

Ví dụ, với `mk01`, `LB = UB = 40`: thuật toán tìm được `C = 42` thì `RPD_BKS = 5%`. Với `mk10`, `LB = 189`, `UB = 193`: tìm được `C = 193` là đạt BKS, nhưng chưa xác nhận tối ưu.

Riêng khi đối chiếu BKS makespan, xét thành phần `C_max` của từng lời giải sau khi giải mã. Khi kết thúc một lần chạy, có thể lấy makespan nhỏ nhất trong các lịch khả thi đã tìm được để so sánh với BKS; kết quả này chưa đánh giá toàn bộ tập nghiệm ba mục tiêu. Trước khi tính RPD, cần kiểm tra lịch thỏa điều kiện chọn máy, thứ tự công đoạn và không chồng lấn trên cùng máy.

Khi chạy nhiều seed, nên ghi `C_best`, `C_mean ± std`, RPD tương ứng, số lần đạt BKS, thời gian chạy, cấu hình máy và số lần chạy. Nếu tổng hợp nhiều instance, tính RPD riêng cho từng instance trước khi lấy trung bình. BKS là mốc chất lượng lời giải, không tự tạo ra một phép so sánh công bằng về tốc độ chạy giữa các thuật toán.
