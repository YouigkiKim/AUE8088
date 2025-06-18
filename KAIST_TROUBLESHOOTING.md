# KAIST RGBT Dataset Troubleshooting Guide

KAIST RGBT 데이터셋에서 발생하는 일반적인 문제들과 해결방법입니다.

## 🔍 문제 진단

### 1. 경로 문제 진단

**증상**: `datasets/kaist-rgbt/train/images/{}/set05_V000_I02007.jpg: No such file or directory`

**원인**: KAIST 데이터셋의 특별한 경로 구조 (`{}` 플레이스홀더)

**해결방법**:
```bash
# 데이터셋 구조 확인
ls datasets/kaist-rgbt/train/images/
# 예상 출력: lwir/  visible/

# 라벨 파일 확인
ls datasets/kaist-rgbt/train/labels/ | head -5
```

### 2. 라벨 형식 문제

**증상**: `AssertionError` in dataloaders.py line 614

**원인**: KAIST 라벨은 6컬럼 (class, x, y, w, h, occlusion), YOLO는 5컬럼

**해결방법**: 전용 데이터로더 사용 (자동으로 처리됨)

## 🚀 빠른 해결책

### 1. KAIST 전용 테스트 스크립트 실행

```bash
python test_kaist_anchor.py --quick
```

이 스크립트는 KAIST 데이터셋에 특화되어 있으며 다음을 수행합니다:
- 전용 데이터로더 사용
- 6컬럼 라벨 형식 처리
- 빠른 앵커 튜닝 테스트 (100세대)

### 2. 데이터셋 경로 확인

```python
# 데이터셋 구조 확인 스크립트
import yaml
with open('data/kaist-rgbt_test.yaml', 'r') as f:
    data = yaml.safe_load(f)
print("Dataset path:", data['path'])
print("Train files:", data['train'])
```

### 3. 라벨 파일 검증

```python
# 라벨 형식 확인
import numpy as np
label_file = 'datasets/kaist-rgbt/train/labels/set00_V000_I00000.txt'
with open(label_file, 'r') as f:
    labels = [x.split() for x in f.read().strip().splitlines()]
labels = np.array(labels, dtype=np.float32)
print(f"Label shape: {labels.shape}")  # (N, 6) 예상
print(f"First label: {labels[0] if len(labels) > 0 else 'No labels'}")
```

## 🔧 단계별 해결

### Step 1: 환경 확인

```bash
# 필요 패키지 확인
python -c "import torch, yaml, numpy, matplotlib; print('✅ All packages available')"

# 데이터셋 존재 확인
ls datasets/kaist-rgbt/
```

### Step 2: 데이터셋 설정 확인

```bash
# YAML 파일 확인
cat data/kaist-rgbt_test.yaml

# 경로가 올바른지 확인
python -c "
import yaml
from utils.general import check_dataset
with open('data/kaist-rgbt_test.yaml', 'r') as f:
    data = yaml.safe_load(f)
try:
    data = check_dataset(data)
    print('✅ Dataset paths OK')
except Exception as e:
    print(f'❌ Dataset error: {e}')
"
```

### Step 3: 데이터로더 테스트

```bash
python test_kaist_anchor.py --quick
```

## 📝 일반적인 오류와 해결책

### 오류 1: "No labels found in cache"

**해결책**:
```bash
# 캐시 파일 삭제 후 재생성
rm datasets/kaist-rgbt/train*.cache
python test_kaist_anchor.py --quick
```

### 오류 2: "Image Not Found"

**해결책**:
```bash
# 이미지 파일 존재 확인
find datasets/kaist-rgbt -name "*.jpg" | head -5

# 경로 구조 확인
tree datasets/kaist-rgbt -L 3
```

### 오류 3: "AssertionError in dataloaders.py"

**해결책**: KAIST 전용 데이터로더 사용 (이미 수정됨)

## 🎯 권장 사용법

### 1. 빠른 테스트 (추천)

```bash
# KAIST 전용 스크립트로 빠른 테스트
python test_kaist_anchor.py --quick
```

### 2. 전체 검증

```bash
# 메인 스크립트로 전체 검증 (시간 오래 걸림)
python test_anchor_tuning.py --data data/kaist-rgbt_test.yaml --class-id 0 --gen 100
```

### 3. 다른 클래스 테스트

```bash
# Cyclist 클래스 (클래스 1)
python test_anchor_tuning.py --data data/kaist-rgbt_test.yaml --class-id 1 --gen 100
```

## 🔄 데이터셋 준비 확인사항

### 필수 디렉토리 구조:
```
datasets/kaist-rgbt/
├── train/
│   ├── images/
│   │   ├── lwir/
│   │   └── visible/
│   └── labels/
├── val/
└── test-all-20.txt
```

### 필수 파일:
- `train-D.txt`: 트레이닝 이미지 목록
- `test-all-20.txt`: 테스트 이미지 목록
- 라벨 파일들 (6컬럼 형식)

## 💡 추가 팁

### 1. 메모리 절약
```bash
# 작은 배치 사이즈 사용
python test_anchor_tuning.py --data data/kaist-rgbt_test.yaml --class-id 0 --gen 50
```

### 2. 디버깅 모드
```bash
# 상세 로그 출력
python test_kaist_anchor.py --quick 2>&1 | tee debug.log
```

### 3. 가상환경 사용
```bash
# 깨끗한 환경에서 테스트
conda create -n kaist_test python=3.8
conda activate kaist_test
pip install -r requirements.txt
python test_kaist_anchor.py --quick
```

## 📞 추가 도움이 필요하면

1. **빠른 테스트**: `python test_kaist_anchor.py --quick`
2. **로그 확인**: 오류 메시지의 전체 내용 확인
3. **데이터셋 구조**: `tree datasets/kaist-rgbt -L 3` 실행 결과 확인 