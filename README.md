# RadarandVision
# 🛰️ Multi-modal Fusion for UAV Object Detection
> **mmWave Radar + RGB Image Fusion using Attention-based Mid-level Architecture**

[cite_start]이 프로젝트는 UAV(무인항공기) 환경에서 객체 탐지 성능을 극대화하기 위해 **mmWave 레이더(Range-Doppler Map)**와 **RGB 영상** 데이터를 결합하는 경량 딥러닝 모델을 연구합니다. 

## 📌 1. 연구 목적 및 필요성 (Objective & Necessity)
* [cite_start]**기존의 한계:** 영상 기반 탐지는 조도 및 기상 변화에 취약하며, 레이더는 객체의 형태 정보가 부족합니다. 
* [cite_start]**연구 목표:** 두 센서의 장점을 결합하여 환경 변화에 강건(Robust)하면서도, Jetson NX와 같은 엣지 디바이스에서 실시간 구동 가능한 경량 모델을 설계합니다. 

## 🧠 2. 주요 제안 방법 (Proposed Method)
[cite_start]이 연구는 AI의 잠재력을 극대화하기 위해 구조화된 고급 테크닉을 적용합니다. [cite: 12, 13]

* **Data Generation:** MATLAB 기반 시뮬레이션을 통해 동기화된 Radar RD Map(64x64)과 RGB Image를 생성합니다.
* **Architecture:** YOLOv8n을 백본으로 하며, **Attention 메커니즘**을 활용한 **Mid-level Feature Fusion**을 수행합니다.
* **Optimization:** 엣지 디바이스 배포를 위한 모델 경량화 및 실시간 처리 최적화를 목표로 합니다.

## 📅 3. 추진 일정 (Project Schedule)
[cite_start]석사 학위 취득을 위한 12개월간의 체계적인 연구 로드맵입니다. [cite: 56]

| 단계 | 기간 | 주요 과업 |
|:---:|:---:|:---|
| **Phase 1** | 1-4월 | [cite_start]선행 연구 조사 및 MATLAB 데이터셋 구축 파이프라인 완성 [cite: 56] |
| **Phase 2** | 5-8월 | [cite_start]Attention Fusion 모델 구현 및 비교 실험 수행 (Single vs Fusion) [cite: 56] |
| **Phase 3** | 9-12월 | [cite_start]Jetson NX 최적화 및 최종 학위 논문 집필 [cite: 56] |

## 🛠️ 4. 기술 스택 (Tech Stack)
* [cite_start]**Language:** Python, MATLAB 
* **Framework:** PyTorch, Ultralytics (YOLOv8)
* **Environment:** Jetson NX, Ubuntu 20.04

## 📂 5. 프로젝트 구조 (File Structure)
[cite_start]프롬프트 전문가를 위한 특별 자료 모음처럼 정돈된 구조를 지향합니다. [cite: 16]
* `data_gen.m`: MATLAB 기반 합성 데이터 생성 스크립트
* `models/`: Attention Fusion 모델 정의 코드
* [cite_start]`utils/`: 데이터 로더 및 전처리 유틸리티 [cite: 43]
* `train.py`: 모델 학습 및 검증 파이프라인

---
**Contact:** 
