# 🏗️ MLOps 기반 의료 AI 프로젝트 표준 구조

## 📋 개요
모든 의료 AI 프로젝트에 적용할 수 있는 표준화된 MLOps 디렉토리 구조입니다. 이 구조는 실험 관리, 모델 배포, 서비스 운영을 위한 최소한의 필수 요소들을 포함합니다.

---

## 📁 표준 프로젝트 구조

```
medical-ai-project/
├── 📁 src/                          # 소스 코드
│   ├── 📁 api/                      # FastAPI 서버
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI 앱 메인
│   │   ├── routes/                  # API 라우트
│   │   │   ├── __init__.py
│   │   │   ├── health.py           # 헬스체크 엔드포인트
│   │   │   └── predict.py          # 추론 엔드포인트
│   │   ├── middleware/              # 미들웨어
│   │   │   ├── __init__.py
│   │   │   ├── auth.py             # 인증 미들웨어
│   │   │   └── logging.py          # 로깅 미들웨어
│   │   └── utils/                   # API 유틸리티
│   │       ├── __init__.py
│   │       ├── response.py         # 응답 포맷
│   │       └── validation.py       # 입력 검증
│   ├── 📁 models/                   # AI 모델 (코드)
│   │   ├── __init__.py
│   │   ├── base.py                 # 기본 모델 클래스
│   │   └── [project_name].py       # 프로젝트별 모델
│   ├── 📁 utils/                    # 유틸리티
│   │   ├── __init__.py
│   │   ├── preprocessing.py        # 전처리 함수
│   │   ├── visualization.py        # 시각화 함수
│   │   ├── metrics.py              # 평가 지표
│   │   └── medical_utils.py        # 의료 특화 유틸리티
│   └── 📁 mlops/                    # MLOps 도구 (선택)
│       ├── __init__.py
│       ├── model_registry.py       # 모델 레지스트리
│       ├── experiment_tracker.py   # 실험 추적
│       └── deployment.py           # 배포 관리
├── 📁 configs/                      # 설정 파일
│   ├── base.yaml                   # 기본 설정
│   ├── development.yaml            # 개발 환경
│   ├── production.yaml             # 운영 환경
│   ├── model_configs/              # 모델별 설정
│   │   └── [project_name].yaml     # 프로젝트별 모델 설정
│   └── api_configs/                # API 설정
│       ├── fastapi.yaml
│       └── docker.yaml
├── 📁 data/                         # 데이터 (gitignore)
│   ├── 📁 raw/                     # 원본 데이터
│   ├── 📁 processed/               # 전처리된 데이터
│   ├── 📁 samples/                 # 샘플 데이터 (테스트용)
│   └── 📁 external/                # 외부 데이터
├── 📁 docker/                       # Docker 설정
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml      # 개발용
│   ├── docker-compose.prod.yml     # 운영용
│   └── 📁 scripts/
│       ├── build.sh
│       ├── run.sh
│       └── deploy.sh
├── 📁 tests/                        # 테스트 코드
│   ├── __init__.py
│   ├── 📁 unit/                    # 단위 테스트
│   │   ├── test_models.py
│   │   ├── test_preprocessing.py
│   │   └── test_api.py
│   ├── 📁 integration/             # 통합 테스트
│   │   ├── test_end_to_end.py
│   │   └── test_deployment.py
│   └── 📁 fixtures/                # 테스트 데이터
│       ├── sample_images/
│       └── sample_models/
├── 📁 docs/                         # 문서
│   ├── README.md                   # 프로젝트 소개
│   ├── API.md                      # API 문서
│   ├── DEPLOYMENT.md               # 배포 가이드
│   ├── DEVELOPMENT.md              # 개발 가이드
│   ├── MEDICAL_AI.md               # 의료 AI 특화 가이드
│   └── 📁 images/                  # 문서용 이미지
├── 📁 scripts/                      # 실행 스크립트
│   ├── train.py                    # 모델 학습
│   ├── evaluate.py                 # 모델 평가
│   ├── predict.py                  # 모델 추론
│   ├── deploy.py                   # 배포
│   └── 📁 setup/                   # 환경 설정
│       ├── install_dependencies.sh
│       └── setup_environment.sh
├── 📁 notebooks/                    # Jupyter 노트북 (선택사항)
│   ├── 📁 exploration/             # 데이터 탐색
│   ├── 📁 experiments/             # 실험 노트북
│   └── 📁 tutorials/               # 튜토리얼
├── .env.example                     # 환경 변수 예시
├── .gitignore                       # Git 무시 파일
├── requirements.txt                 # Python 의존성
├── requirements-dev.txt             # 개발용 의존성
├── setup.py                         # 패키지 설정
├── pyproject.toml                   # 프로젝트 메타데이터
├── Makefile                         # 빌드 자동화
├── README.md                        # 프로젝트 메인 문서
└── mlruns/                          # MLflow 실험 결과 자동 생성(학습 후)
```

---

## 🎯 각 디렉토리 설명

### **src/**
- **api/**: FastAPI 서버 및 API 엔드포인트
- **models/**: AI 모델 정의 및 구현 (코드)
- **utils/**: 공통 유틸리티 함수
- **mlops/**: MLOps 도구 및 관리 (선택)

### **configs/**
- 환경별/모델별/API별 설정 파일

### **data/**
- 원본, 전처리, 샘플, 외부 데이터 (gitignore)

### **docker/**
- Dockerfile, docker-compose 등 컨테이너화 설정

### **tests/**
- 단위/통합 테스트, 테스트용 데이터

### **docs/**
- 프로젝트 문서, API/배포/개발 가이드, 이미지

### **scripts/**
- 학습/추론/배포/평가 스크립트, 환경설정

### **notebooks/**
- Jupyter 노트북 (선택사항)

### **mlruns/**
- MLflow 실험 결과 자동 생성 디렉토리 (학습 시 자동 생성, 직접 관리 불필요)

---

## ❌ 더 이상 포함하지 않는 디렉토리
- **models/**: 학습된 모델 직접 저장 X (MLflow로 일원화)
- **mlflow/**: MLflow는 mlruns/에 자동 저장, 별도 디렉토리 불필요

---

이 표준 구조를 기반으로 모든 의료 AI 프로젝트를 일관되게 구성하여 **유지보수성**, **확장성**, **재사용성**을 극대화할 수 있습니다. 