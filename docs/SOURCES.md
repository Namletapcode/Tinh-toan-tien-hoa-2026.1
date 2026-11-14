# Nguồn mã và giấy phép

[README](../README.md) · [Nguồn dữ liệu và BKS](BENCHMARKS.md)

| Nguồn | Commit cố định | Vai trò |
| --- | --- | --- |
| [moead-framework/framework](https://github.com/moead-framework/framework) | `82e437678f5d54a13a71b8a97fa9a45cb6ec4685` | Khung MOEA/D và các giao diện component |
| [Aihong-Sun/MOEA-D-and-NSGA--for-FJSP](https://github.com/Aihong-Sun/MOEA-D-and-NSGA--for-FJSP) | `3318367c7c5f9b355c3ff0426963c759e806e753` | Tham khảo OS/MS, decoder và ba mục tiêu FJSP |

Mã dùng để chạy nằm trong `vendor/moead_framework/`, là bản phát triển của
project. Phần FJSP viết riêng theo giao diện framework; hai repo tham khảo
không phải dependency runtime. Mô hình và hành vi hiện tại nằm ở [FJSP](FJSP.md).

Giấy phép framework là **GNU GPL v3**, lưu tại [vendor/LICENCE](../vendor/LICENCE).
Giữ thông tin nguồn và giấy phép đi kèm mã. `vendor/README.md` là tài liệu
upstream; hướng dẫn sử dụng project nằm ở README gốc.

`references/moead-framework/` và `references/aih-sun-fjsp/` là bản đối chiếu
tùy chọn, được bỏ qua trong Git. Khi cần lấy lại, chạy từ gốc project nếu các
thư mục tương ứng chưa tồn tại:

```powershell
git clone https://github.com/moead-framework/framework.git references/moead-framework
git -C references/moead-framework checkout --detach 82e437678f5d54a13a71b8a97fa9a45cb6ec4685
git clone https://github.com/Aihong-Sun/MOEA-D-and-NSGA--for-FJSP.git references/aih-sun-fjsp
git -C references/aih-sun-fjsp checkout --detach 3318367c7c5f9b355c3ff0426963c759e806e753
```

Commit trên xác định nguồn xuất phát, không xác định toàn bộ mã hiện tại.
Metadata `version=1.0` trong `vendor/setup.py` thuộc upstream; mốc nghiên cứu
của nhóm được quản lý riêng bằng Git. Manifest và fingerprint của các lượt
chạy được mô tả trong [EXPERIMENTS](EXPERIMENTS.md).
