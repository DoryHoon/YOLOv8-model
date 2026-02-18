# YOLOv8 KITTI 객체 탐지 모델 - PyTorch 구현

---

## 1. YOLOv8이란?

YOLOv8은 Ultralytics에서 개발한 실시간 객체 탐지(Object Detection) 모델 시리즈의 최신 버전입니다. YOLO(You Only Look Once)는 이미지를 단 한 번만 처리하여 객체의 위치와 종류를 동시에 예측하는 방식으로, 빠른 속도와 높은 정확도를 동시에 달성합니다.

YOLOv8의 주요 특징은 다음과 같습니다:

- **앵커 프리(Anchor-free):** 이전 버전과 달리 사전 정의된 앵커 박스 없이 객체의 중심점을 직접 예측합니다. 이를 통해 하이퍼파라미터 튜닝 부담이 줄고 일반화 성능이 향상됩니다.
- **분리된 탐지 헤드(Decoupled Head):** 분류(Classification)와 박스 회귀(Bounding Box Regression) 브랜치를 분리하여 각각 최적화합니다.
- **DFL 손실(Distribution Focal Loss):** 경계 박스 좌표를 단일 값이 아닌 확률 분포로 예측하여 더 정밀한 위치 추정을 가능하게 합니다.
- **다양한 모델 크기:** n(nano), s(small), m(medium), l(large), x(extra-large) 5가지 버전을 제공하여 성능과 속도의 균형을 선택할 수 있습니다.

---

## 2. 전체 아키텍처 구조

YOLOv8은 세 가지 주요 모듈로 구성됩니다: **Backbone → Neck → Head**

```
입력 이미지 (640×640×3)
        |
        v
  ┌─────────────┐
  │  Backbone   │  특징 추출 (Feature Extraction)
  │  Conv, C2f, │  → P3 (80×80), P4 (40×40), P5 (20×20) 출력
  │  SPPF       │
  └─────────────┘
        |
        v
  ┌─────────────┐
  │    Neck     │  특징 융합 (Feature Fusion, FPN+PAN)
  │  Upsample + │  → 다중 스케일 특징을 하나로 통합
  │  C2f        │
  └─────────────┘
        |
        v
  ┌─────────────┐
  │    Head     │  예측 (Prediction)
  │  Detect     │  → Bbox(위치) + Cls(분류) 예측
  │  DFL + BCE  │
  └─────────────┘
        |
        v
탐지 결과: [클래스, 신뢰도, x, y, w, h]
```

---

## 3. 핵심 구성 요소 (Building Blocks)

### 3-1. Conv — 기본 합성곱 블록

YOLOv8의 가장 기본 단위입니다. Conv2d → BatchNorm2d → SiLU 활성화 함수를 순서대로 적용합니다.

SiLU(Sigmoid Linear Unit)는 ReLU보다 부드러운 비선형 활성화 함수로, 학습 안정성과 성능을 향상시킵니다.

```python
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1, activation=True):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

### 3-2. Bottleneck — 잔차 연결 블록

두 개의 Conv 레이어와 선택적 숏컷(shortcut) 연결로 구성됩니다. `shortcut=True`일 때 입력과 출력을 더하는 잔차 연결(Residual Connection)이 활성화되어 기울기 소실 문제를 완화하고 학습을 안정시킵니다.

```python
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return (x + self.conv2(self.conv1(x))) if self.add else self.conv2(self.conv1(x))
```

### 3-3. C2f — Cross-Stage Partial Bottleneck (핵심 블록)

YOLOv8의 가장 핵심적인 블록입니다. 입력 특징맵을 두 부분으로 분리(Split)한 뒤, 한 부분은 여러 Bottleneck을 거치고 모든 중간 출력을 Concat하여 다양한 수준의 특징을 동시에 활용합니다. 이를 통해 높은 표현력을 유지하면서 계산 효율도 높입니다.

```python
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=False, expansion=0.5):
        self.c = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, 2*self.c, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut) for _ in range(num_bottlenecks)])
        self.cv2 = Conv((2 + num_bottlenecks) * self.c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        y = list(x.split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

### 3-4. SPPF — Spatial Pyramid Pooling Fast

다양한 크기의 수용 영역(receptive field)을 효율적으로 통합하는 블록입니다. MaxPool2d를 순차적으로 3번 적용한 결과와 원본 특징을 모두 Concat하여, 작은 객체부터 큰 객체까지 다양한 스케일에 대응하는 특징을 생성합니다.

```python
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.conv2 = Conv(4 * out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), 1))
```

---

## 4. 세부 모듈 설명

### 4-1. Backbone — 특징 추출기

Backbone은 입력 이미지에서 다중 스케일 특징을 추출하는 역할을 합니다. Conv와 C2f 블록을 쌓아 점진적으로 공간 해상도를 줄이고 채널 수를 늘리며, 마지막에 SPPF로 다양한 스케일 정보를 통합합니다.

Backbone의 출력은 세 가지 스케일의 특징맵으로, 이를 Neck에 전달합니다:
- **P3 (C2f_out_4):** 80×80 — 작은 객체 탐지에 적합
- **P4 (C2f_out_6):** 40×40 — 중간 객체 탐지에 적합
- **P5 (SPPF_out_9):** 20×20 — 큰 객체 탐지에 적합

모델 크기에 따라 depth(d), width(w), ratio(r) 파라미터가 다르게 설정됩니다:

| 버전 | depth (d) | width (w) | ratio (r) |
|------|-----------|-----------|-----------|
| n (nano) | 0.33 | 0.25 | 2.0 |
| s (small) | 0.33 | 0.50 | 2.0 |
| m (medium) | 0.67 | 0.75 | 1.5 |
| l (large) | 1.00 | 1.00 | 1.0 |
| x (extra) | 1.00 | 1.25 | 1.0 |

### 4-2. Neck — 특징 융합기 (FPN + PAN)

Neck은 Backbone에서 추출된 다중 스케일 특징을 융합합니다. Top-Down Path(큰 스케일 → 작은 스케일, Upsample 사용)와 Bottom-Up Path(작은 스케일 → 큰 스케일, Conv 사용)를 결합하는 FPN+PAN 구조를 사용합니다. 이를 통해 각 스케일에서 얕은 특징(위치 정보)과 깊은 특징(의미 정보)이 모두 반영됩니다.

```
Top-Down:  P5 → Upsample → Concat(P4) → C2f
                         → Upsample → Concat(P3) → C2f

Bottom-Up: C2f → Conv(stride=2) → Concat → C2f
               → Conv(stride=2) → Concat → C2f
```

### 4-3. Head — 예측기 (Detect + DFL)

Head는 분류(Cls)와 경계 박스 회귀(Bbox) 두 개의 독립된 브랜치로 구성됩니다.

**DFL (Distribution Focal Loss):** 경계 박스 좌표(x, y, w, h)를 단일 숫자가 아닌 16개 빈(bin)의 확률 분포로 예측합니다. 예를 들어 x 좌표를 예측할 때, 모델은 0~1 범위를 16등분한 각 구간에 해당할 확률을 출력하고, 이를 가중 평균하여 최종 좌표를 계산합니다. 이 방식은 불확실성이 높은 경계에서도 더 정밀한 좌표 추정이 가능합니다.

**손실 함수:**
- 경계 박스: CIoU + DFL
- 분류: BCE (Binary Cross Entropy)

이 프로젝트는 KITTI 데이터셋 기준 `nc=8` (Car, Van, Truck, Pedestrian, Person sitting, Cyclist, Tram, Misc)로 설정되어 있습니다.

---

## 5. 전체 모델 구성 및 실행 흐름

세 모듈을 하나로 조합한 최종 모델 클래스입니다.

```python
class YOLOv8_KITTI(nn.Module):
    def __init__(self, version='s', nc=8):
        # 1. Backbone: 특징 추출
        self.backbone = Backbone(version)

        # 2. Neck: 다중 스케일 특징 융합
        self.neck = Neck(version)

        # 3. Detect Head: 분류 및 경계 박스 예측
        ch = [int(256*w), int(512*w), int(512*w*r)]
        self.detect = Detect(nc=nc, ch=ch)

    def forward(self, x):
        # Backbone → Neck → Head 순서로 순전파
        features = self.backbone(x)               # P3, P4, P5 출력
        neck_out = self.neck(None, *features)      # 특징 융합
        return self.detect(list(neck_out))         # 최종 예측
```

**실행 흐름 요약:**

```
입력 이미지
    │
    ▼
Backbone.forward(x)
    → conv_0 ~ conv_7, c2f_2 ~ c2f_8, sppf
    → 반환: (C2f_out_4, C2f_out_6, SPPF_out_9)
    │
    ▼
Neck.forward(x, C2f_out_4, C2f_out_6, SPPF_out_9)
    → Top-down: Upsample + Concat + C2f
    → Bottom-up: Conv(stride=2) + Concat + C2f
    → 반환: (C2f_out_15, C2f_out_18, C2f_out_21)
    │
    ▼
Detect.forward([C2f_out_15, C2f_out_18, C2f_out_21])
    → 각 스케일별 bbox_branch + cls_branch
    → DFL로 좌표 정제 (추론 시)
    → 반환: 각 스케일별 [Bbox, Cls] 텐서
```

---

## 6. 프로젝트 구조

```
.
├── YOLOv8_FULL_Implementation.ipynb   # 전체 구현 노트북
├── YOLOv8_Architecture_diagram.pdf    # 아키텍처 다이어그램
└── README.md                          # 이 파일
```

---

## 7. 참고 사항

- 본 구현은 KITTI 데이터셋(8개 클래스)을 기준으로 설계되었습니다.
- 모델 크기는 `version` 인자(`'n'`, `'s'`, `'m'`, `'l'`, `'x'`)로 선택할 수 있습니다.
- Ultralytics의 공식 YOLOv8을 학습 목적으로 처음부터 재구현한 코드입니다.
