# Anchor Tuning Validation Tool

Person 클래스에 특화된 앵커 튜닝 알고리즘을 검증하는 도구입니다.

## 📋 기능

- **클래스별 라벨 분석**: 전체 클래스 vs Person 클래스 라벨 분포 분석
- **앵커 최적화**: Person 클래스에 특화된 k-means + 유전 알고리즘 앵커 튜닝
- **성능 비교**: 기존 앵커 vs 새 앵커의 fitness 점수 비교
- **시각화**: 라벨 분포와 앵커 비교 그래프 생성
- **상세 리포트**: 검증 결과를 텍스트 파일로 저장

## 🚀 사용법

### 1. 간편 실행 (추천)

```bash
python run_anchor_validation.py
```

대화형 인터페이스로 데이터셋과 모델을 선택할 수 있습니다.

### 2. 직접 실행

```bash
python test_anchor_tuning.py --data data/your_dataset.yaml [옵션들]
```

#### 필수 파라미터
- `--data`: 데이터셋 YAML 파일 경로

#### 선택 파라미터
- `--model`: 모델 YAML 파일 경로 (기존 앵커 비교용)
- `--class-id`: 타겟 클래스 ID (기본값: 0=person)
- `--img-size`: 이미지 크기 (기본값: 640)
- `--anchors`: 앵커 개수 (기본값: 9)
- `--gen`: 유전 알고리즘 세대 수 (기본값: 1000)
- `--save-dir`: 결과 저장 디렉토리 (기본값: runs/anchor_validation)

### 3. 예제 명령어

```bash
# COCO 데이터셋으로 Person 클래스 앵커 튜닝
python test_anchor_tuning.py --data data/coco.yaml --model models/yolov5s.yaml

# nuScenes 데이터셋으로 빠른 테스트 (100세대)
python test_anchor_tuning.py --data data/nuscenes.yaml --gen 100

# 다른 클래스 (예: car=2) 앵커 튜닝
python test_anchor_tuning.py --data data/custom.yaml --class-id 2
```

## 📊 출력 결과

### 1. 콘솔 출력
- 데이터셋 로딩 정보
- 라벨 분포 통계
- k-means 진행상황
- 앵커 fitness 비교 결과

### 2. 시각화 파일
`runs/anchor_validation/anchor_validation.png`
- 상단 좌측: 전체 클래스 라벨 분포
- 상단 우측: Person 클래스 라벨 분포
- 하단 좌측: 기존 앵커
- 하단 우측: 새로 생성된 앵커

### 3. 결과 파일
`runs/anchor_validation/validation_results.txt`
- 실행 파라미터
- 라벨 통계
- 앵커 성능 비교
- 최종 앵커 값

## 🔍 결과 해석

### Fitness 점수
- **높을수록 좋음** (0~1 범위)
- 앵커와 라벨의 width/height 비율 매칭 정도
- 0.6 이상이면 양호, 0.8 이상이면 우수

### 성능 지표
```
Original anchors fitness (person only): 0.6234
New anchors fitness (person only): 0.7891
Person class improvement: +0.1657 (+26.6%)
```

### 라벨 분포 분석
```
Total labels: 12345
Class 0 labels: 3456 (28.0%)
```

## ⚙️ 설정 및 커스터마이징

### 클래스 ID 확인
데이터셋의 클래스 ID를 확인하려면:
```python
import yaml
with open('data/your_dataset.yaml', 'r') as f:
    data = yaml.safe_load(f)
print(data['names'])  # 클래스 이름 리스트
```

### 다른 클래스 적용
```bash
# Car 클래스가 2번인 경우
python test_anchor_tuning.py --data data/custom.yaml --class-id 2
```

### 빠른 테스트
```bash
# 100세대로 빠른 테스트
python test_anchor_tuning.py --data data/your_dataset.yaml --gen 100
```

## 🐛 문제 해결

### 일반적인 오류

1. **"No labels found for class X"**
   - 해당 클래스의 라벨이 데이터셋에 없음
   - 클래스 ID를 확인하거나 다른 클래스 시도

2. **"Dataset path not found"**
   - 데이터셋 경로 확인
   - YAML 파일 내 train 경로 확인

3. **"Memory error during k-means"**
   - 더 작은 데이터셋 사용
   - `--gen` 값을 줄여서 시도

### 성능 최적화

- **빠른 테스트**: `--gen 100`
- **정확한 결과**: `--gen 1000` (기본값)
- **최고 정확도**: `--gen 3000`

## 📈 활용 예시

### 1. Person Detection 최적화
```bash
python test_anchor_tuning.py --data data/pedestrian_dataset.yaml --class-id 0
```

### 2. Vehicle Detection 최적화
```bash
python test_anchor_tuning.py --data data/traffic_dataset.yaml --class-id 2
```

### 3. 다중 클래스 비교
각 클래스별로 실행하여 어떤 클래스가 가장 큰 개선을 보이는지 확인

## 🔧 고급 사용법

### 커스텀 앵커 수
```bash
# 6개 앵커 (P3, P4 각각 3개)
python test_anchor_tuning.py --data data/custom.yaml --anchors 6

# 12개 앵커 (P3, P4, P5, P6 각각 3개)
python test_anchor_tuning.py --data data/custom.yaml --anchors 12
```

### 배치 실행 스크립트
```python
# batch_validation.py
import subprocess
classes = [0, 1, 2]  # person, bicycle, car
for cls in classes:
    cmd = f"python test_anchor_tuning.py --data data/custom.yaml --class-id {cls} --save-dir runs/class_{cls}"
    subprocess.run(cmd.split())
```

## 📝 참고사항

- **실행 시간**: 데이터셋 크기와 세대 수에 따라 수분~수십분 소요
- **메모리 요구사항**: 대용량 데이터셋의 경우 8GB+ RAM 권장
- **Python 버전**: Python 3.7 이상 필요
- **의존성**: matplotlib, numpy, torch, scipy 필요 