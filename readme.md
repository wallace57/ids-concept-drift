## BTQT - Mô phỏng sự suy giảm mô hình IDS và đề xuất khắc phục trước Concept drift của NSL-KDD

- **Notebook 00 & 01** phản ánh yêu cầu của bài tập 01 trong slide (`00_eda.ipynb` và `01_stimulate_CF_full.ipynb`)
- **Notebook 02** phản ánh yêu cầu bài tập 02 trong slide (`02_stimulate_CF_phases.ipynb`)
- **Notebook 03** phản ánh yêu cầu bài tập 03 trong slide (`03_solution_ARFs.ipynb`)

## References

- https://inseclab.uit.edu.vn/nsl-kdd-goc-nhin-chi-tiet-ve-tap-du-lieu-huan-luyen-cho-cac-ids/
- https://github.com/thinline72/nsl-kdd
- https://www.kaggle.com/code/mihirs16/arf-fy-project

---

# IDS Concept Drift - Nghiên cứu và Khắc phục Suy giảm Hiệu suất

Dự án nghiên cứu về **concept drift** trong hệ thống **Intrusion Detection System (IDS)** sử dụng dataset NSL-KDD. So sánh hiệu quả giữa mô hình tĩnh (Static) và mô hình adaptive (ARF - Adaptive Random Forest) trong việc khắc phục suy giảm hiệu suất do concept drift.

## 📋 Requirements

- **Python**: >= 3.12 (xem `pyproject.toml`, `.python-version`)
- **Core**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- **ARF/Online learning**: `river` (khuyến nghị)
- **Replay CL script** (`replay_cl.py`): `torch`, `packaging`

## 🚀 Hướng dẫn cài đặt

### Bước 1: Clone repository
```bash
git clone <repo-url>
cd ids-concept-drift
```

### Bước 2: Virtual environment (khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
# Cho ARF và notebooks:
pip install river notebook
pip install torch
```

Hoặc dùng uv/pyproject:
```bash
uv sync
```

### Bước 4: Download NSL-KDD Dataset

**Option 1: Script (khuyến nghị)**
```bash
python scripts/download_nsl_kdd.py
```
Script tải `KDDTrain+.txt` và `KDDTest+.txt` từ [GitHub NSL-KDD](https://github.com/thinline72/nsl-kdd) vào folder `data/`.

**Option 2: Manual**
1. Truy cập: https://www.unb.ca/cic/datasets/nsl.html
2. Download `KDDTrain+.txt` và `KDDTest+.txt`
3. Đặt vào folder `data/`

### Bước 5: Chạy

**Workflow chính (Notebooks):**
- Mở và chạy theo thứ tự: `00_eda.ipynb` → `01_stimulate_CF_full.ipynb` → `02_stimulate_CF_phases.ipynb` → `03_solution_ARFs.ipynb`

**Scripts phụ:**
```bash
python replay_cl.py
python exp/ids_concept_drift_ARF_new_v2_river_full.py
```

## 📁 Cấu trúc Project

```
ids-concept-drift/
│
├── 00_eda.ipynb              # EDA NSL-KDD, so sánh label mapping
├── 01_stimulate_CF_full.ipynb   # Task 01: Drift toàn cục – chứng minh suy giảm IDS tĩnh
├── 02_stimulate_CF_phases.ipynb # Task 02: Catastrophic forgetting theo phases
├── 03_solution_ARFs.ipynb       # Task 03: ARF variants
├── 03_solution_improved.ipynb   # Task 03: ARF improved (logging, cấu trúc rõ ràng)
│
├── replay_cl.py              # Continual learning: Baseline vs Replay Buffer (PyTorch)
├── requirements.txt
├── pyproject.toml
├── readme.md
│
├── scripts/
│   └── download_nsl_kdd.py   # Tải NSL-KDD vào data/
│
├── data/
│   ├── KDDTrain+.txt         # ~125K samples
│   └── KDDTest+.txt          # ~22K samples
│
├── exp/
│   ├── ids_concept_drift_ARF_new_v2_river_full.py  # ARF 7 variants (River)
│   ├── eda_nsl_kdd.ipynb
│   ├── part2_exp_*.ipynb
│   └── README.md
│
├── archive/
│   └── results/              # Kết quả experiments cũ (plots, CSV)
│
│
└── .gitignore
```

## 📓 Notebooks

### `00_eda.ipynb` – Exploratory Data Analysis
- Phân tích cấu trúc NSL-KDD, label mapping (Mapping 1 vs Mapping 2)
- So sánh train/test, các chỉ số thống kê

### `01_stimulate_CF_full.ipynb` – Task 01
- **Mục tiêu**: Chứng minh concept drift bằng suy giảm hiệu năng mô hình tĩnh
- **Model**: Random Forest (scikit-learn) – train 1 lần trên KDDTrain+
- **Test**: KDDTest+ (label shift, covariate shift, zero-day attacks)

### `02_stimulate_CF_phases.ipynb` – Task 02
- **Mục tiêu**: Mô phỏng catastrophic forgetting khi IDS tĩnh gặp concept drift theo phases

### `03_solution*.ipynb` – Task 03
- **Mục tiêu**: Giải pháp ARF (River) với 7 drift detectors
- **Phiên bản**: `03_solution.ipynb`, `03_solution_ARFs.ipynb`, `03_solution_improved.ipynb`
- **Metrics**: AA, FM, BWT

## 🔧 Scripts

### `replay_cl.py` – Continual Learning với Replay Buffer
- **Mô tả**: So sánh Baseline vs Replay Buffer trên NSL-KDD (PyTorch MLP)
- **Tasks**: Theo từng nhóm tấn công (Normal → DoS → Probe → R2L → U2R)
- **Metrics**: AA, FM, BWT
- **Chạy**: `python replay_cl.py`

### `exp/ids_concept_drift_ARF_new_v2_river_full.py`
- **Mô tả**: ARF với 7 variants (None, ADWIN, DDM, PageHinkley, KSWIN, HDDM_A, HDDM_W)
- **Dependencies**: `river`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Chạy**: `python exp/ids_concept_drift_ARF_new_v2_river_full.py`

## 🔬 Metrics Đánh giá

| Metric | Ý nghĩa | Giá trị tốt |
|--------|---------|-------------|
| **AA** (Average Accuracy) | Độ chính xác trung bình trên tất cả periods | Càng cao càng tốt |
| **FM** (Forgetting Measure) | Mức độ "quên" kiến thức cũ | Càng thấp càng tốt |
| **BWT** (Backward Transfer) | Khả năng cải thiện performance trên task cũ nhờ học task mới | Càng cao càng tốt |

## 🔧 Troubleshooting

### Lỗi: File not found (KDDTrain+.txt)
```bash
python scripts/download_nsl_kdd.py
```

### Lỗi: `data_dir` trong notebooks
Một số notebook dùng `data_dir = Path('data/')`. Đảm bảo chạy từ thư mục gốc project. Nếu notebook dùng đường dẫn tuyệt đổi (vd: `H:\tdc_window\...`), sửa lại thành `Path('data/')`.

### Lỗi: River import (DDM, HDDM_A, HDDM_W)
```bash
pip install --upgrade river
```

### Lỗi: `replay_cl.py` – torch not found
```bash
pip install torch
```

## 📚 Tài liệu tham khảo

### Concept Drift & Continual Learning
- **River ML**: https://riverml.xyz/latest/
- **ARF Paper**: Gomes et al. "Adaptive Random Forests for evolving data stream classification"

### NSL-KDD
- **Official**: https://www.unb.ca/cic/datasets/nsl.html
- **GitHub**: https://github.com/thinline72/nsl-kdd
