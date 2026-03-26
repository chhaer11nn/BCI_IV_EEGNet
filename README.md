# EEGNet-based Motor Imagery Classification on BCI Competition IV-2a

BCI Competition IV-2a 데이터셋을 활용한 4-class Motor Imagery 분류 프로젝트입니다. EEGNet 아키텍처와 슬라이딩 윈도우 기반 데이터 증강을 적용하여 within-subject T→E 프로토콜로 평가합니다.

A 4-class Motor Imagery classification project using the BCI Competition IV-2a dataset. It applies an EEGNet architecture with sliding window data augmentation, evaluated under the within-subject T→E protocol.

---

## 프로젝트 구조 | Project Structure

```
BCI_IV/
├── A01T.gdf ~ A09T.gdf          # Training session (raw)
├── A01E.gdf ~ A09E.gdf          # Evaluation session (raw)
├── preprocess.py                 # 전처리 + 슬라이딩 윈도우
├── train_eegnet.py               # EEGNet 학습 & 평가
└── preprocessed/
    ├── A01T_preprocessed.npz     # 전처리 완료 데이터
    └── ...
```

---

## 데이터셋 | Dataset

**BCI Competition IV-2a** (9 subjects, 2 sessions each)

| 항목 | 값 |
|---|---|
| 피험자 수 | 9명 (A01–A09) |
| 클래스 | Left hand, Right hand, Foot, Tongue (4-class) |
| EEG 채널 | 22ch (10-20 system) |
| 샘플링 주파수 | 250 Hz |
| 프로토콜 | T session → train, E session → test |

---

## 전처리 파이프라인 | Preprocessing Pipeline

`preprocess.py`에서 수행하는 전체 과정:

```
Raw GDF
  │
  ├── 채널 리네이밍 (22ch EEG 선택)
  ├── IIR Butterworth BPF (1–45 Hz, 5th order)
  ├── Cue event 추출 (769/770/771/772)
  ├── Artifact trial 제거 (event code 1023)
  │
  ├── Epoching
  │     tmin=-0.2s, tmax=4.0s
  │     baseline correction: (-0.2, 0.0)s
  │     amplitude reject: ±150μV
  │
  ├── MI 구간 crop (1.0–4.0s → 3s = 750 samples)
  ├── Epoch-wise z-score normalization
  │
  └── Sliding Window
        window size: 2.0s (500 samples)
        step size:   0.1s (25 samples)
        → 출력 shape: (N_windows, 1, 22, 500)
```

---

## 모델 | Model Architecture

**EEGNet** (Lawhern et al., 2018)

| 하이퍼파라미터 | 값 |
|---|---|
| F1 (temporal filters) | 8 |
| D (depth multiplier) | 2 |
| F2 (pointwise filters) | 16 |
| Dropout | 0.5 |

```
Input (1, 22, 500)
  → Conv2d temporal (1×125) + BN
  → Depthwise Conv2d (22×1) + BN + ELU + AvgPool(1×4) + Dropout
  → Separable Conv2d (1×16) + Pointwise (1×1) + BN + ELU + AvgPool(1×8) + Dropout
  → Flatten → Linear → 4 classes
```

---

## 학습 설정 | Training Configuration

| 항목 | 값 |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Epochs | 300 (fixed) |
| Loss | CrossEntropyLoss |
| Evaluation | T→E within-subject |

---

## 실행 방법 | How to Run

### 1. 환경 설정

```bash
pip install mne numpy torch matplotlib scikit-learn
```

### 2. 전처리

```bash
python preprocess.py
```

`preprocessed/` 디렉토리에 각 피험자별 `.npz` 파일이 생성됩니다.

### 3. 학습 & 평가

```bash
python train_eegnet.py
```

피험자별 Accuracy, Precision, F1, AUC가 출력되고, Loss/Accuracy 커브가 PNG로 저장됩니다.

---

## 평가 지표 | Evaluation Metrics

각 피험자에 대해 다음 지표를 산출합니다:

- **Accuracy** — 전체 정확도
- **Precision** — macro-averaged 정밀도
- **F1 Score** — macro-averaged F1
- **AUC** — macro-averaged ROC AUC (one-vs-rest)
- **Chance level** — 0.25 (4-class)

---

## 기술 스택 | Tech Stack

`Python` · `PyTorch` · `MNE-Python` · `scikit-learn` · `NumPy` · `Matplotlib`

---

## 참고 문헌 | References

- Lawhern, V. J. et al. (2018). *EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces.* Journal of Neural Engineering, 15(5), 056013.
- Brunner, C. et al. (2008). *BCI Competition IV Dataset 2a.* Graz University of Technology.
