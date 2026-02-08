# Image Classification with LoRA Fine-Tuning

ViT-Large 기반 이미지 분류 모델을 LoRA(Low-Rank Adaptation)로 파인튜닝하는 파이프라인입니다.

## 프로젝트 구조

```
.
├── configs/
│   └── default.yaml          # 학습/평가/추론 설정
├── src/
│   ├── config.py              # 설정 로더 및 CLI 파서
│   ├── dataset.py             # 데이터셋 로딩 및 전처리
│   ├── model.py               # LoRA 모델 구성
│   ├── train.py               # 학습 스크립트
│   ├── evaluate.py            # 평가 스크립트 (메트릭 + 시각화)
│   └── inference.py           # 추론 스크립트 (단일/배치)
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 데이터셋 준비

### 방법 1: 로컬 ImageFolder 형식

```
data/
├── train/
│   ├── cat/
│   │   ├── img001.jpg
│   │   └── ...
│   └── dog/
│       ├── img001.jpg
│       └── ...
└── val/
    ├── cat/
    └── dog/
```

### 방법 2: HuggingFace 데이터셋

`configs/default.yaml`에서 `hf_dataset` 필드를 설정합니다:
```yaml
dataset:
  hf_dataset: "cifar10"   # 또는 "food101", "oxford_flowers102" 등
```

## 사용법

### 1. 학습

```bash
# 기본 설정으로 학습
python -m src.train --config configs/default.yaml --data_dir ./data

# HuggingFace 데이터셋으로 학습
python -m src.train --hf_dataset cifar10

# CLI에서 하이퍼파라미터 오버라이드
python -m src.train --data_dir ./data --num_epochs 20 --batch_size 16 --lora_r 8 --learning_rate 1e-4
```

### 2. 평가

```bash
# 학습된 모델 평가 (best_model 자동 로드)
python -m src.evaluate --config configs/default.yaml --data_dir ./data

# 특정 체크포인트로 평가
python -m src.evaluate --data_dir ./data --checkpoint_path ./outputs/checkpoint-epoch-5
```

출력물:
- `outputs/evaluation/metrics.json` — 전체 메트릭
- `outputs/evaluation/confusion_matrix.png` — 혼동 행렬
- `outputs/evaluation/per_class_metrics.png` — 클래스별 메트릭
- `outputs/evaluation/training_curves.png` — 학습 곡선

### 3. 추론

```bash
# 단일 이미지 추론
python -m src.inference --image_path ./test.jpg

# 폴더 내 전체 이미지 배치 추론
python -m src.inference --image_dir ./test_images/

# 특정 체크포인트 사용
python -m src.inference --checkpoint_path ./outputs/best_model --image_path ./test.jpg
```

## 주요 설정 (configs/default.yaml)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `model.name` | `google/vit-large-patch16-224` | 베이스 모델 |
| `lora.r` | 16 | LoRA rank |
| `lora.lora_alpha` | 32 | LoRA scaling factor |
| `lora.target_modules` | `["query", "value"]` | LoRA 적용 대상 모듈 |
| `training.num_epochs` | 10 | 학습 에포크 수 |
| `training.batch_size` | 32 | 배치 크기 |
| `training.learning_rate` | 5e-4 | 학습률 |
| `training.fp16` | true | Mixed precision 사용 여부 |

## LoRA 장점

- 전체 파라미터의 약 **0.5~2%** 만 학습하여 GPU 메모리 절약
- 원본 모델 가중치를 보존하면서 태스크별 어댑터만 저장
- 빠른 학습 속도와 적은 스토리지 요구
