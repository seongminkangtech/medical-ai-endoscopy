"""
의료 AI 모델의 기본 인터페이스 정의

이 모듈은 모든 의료 AI 모델이 구현해야 하는 기본 메서드들을 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """모든 의료 AI 모델의 기본 클래스
    
    이 클래스는 의료 AI 모델이 구현해야 하는 공통 인터페이스를 정의합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """모델 초기화
        
        Args:
            config: 모델 설정을 담은 딕셔너리
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.model_path = None
        
        # 로깅 설정
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> Any:
        """모델 아키텍처 구축
        
        Args:
            input_shape: 입력 데이터의 형태
            num_classes: 분류할 클래스 수
            
        Returns:
            구축된 모델 객체
        """
        pass
    
    @abstractmethod
    def train(self, train_data: Any, val_data: Any, **kwargs) -> Dict[str, Any]:
        """모델 훈련
        
        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            **kwargs: 추가 훈련 파라미터
            
        Returns:
            훈련 결과를 담은 딕셔너리
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """모델 추론
        
        Args:
            input_data: 추론할 입력 데이터
            
        Returns:
            모델 예측 결과
        """
        pass
    
    @abstractmethod
    def save_model(self, path: Union[str, Path]) -> None:
        """모델 저장
        
        Args:
            path: 모델을 저장할 경로
        """
        pass
    
    @abstractmethod
    def load_model(self, path: Union[str, Path]) -> None:
        """모델 로드
        
        Args:
            path: 모델을 로드할 경로
        """
        pass
    
    def get_model_summary(self) -> str:
        """모델 구조 요약 반환
        
        Returns:
            모델 구조를 설명하는 문자열
        """
        if self.model is None:
            return "모델이 아직 구축되지 않았습니다."
        
        try:
            # TensorFlow 모델인 경우
            if hasattr(self.model, 'summary'):
                from io import StringIO
                summary_io = StringIO()
                self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
                return summary_io.getvalue()
            
            # PyTorch 모델인 경우
            elif hasattr(self.model, 'parameters'):
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                return f"PyTorch 모델\n총 파라미터: {total_params:,}\n훈련 가능 파라미터: {trainable_params:,}"
            
            else:
                return "모델 요약을 생성할 수 없습니다."
                
        except Exception as e:
            self.logger.error(f"모델 요약 생성 실패: {e}")
            return f"모델 요약 생성 실패: {e}"
    
    def is_model_ready(self) -> bool:
        """모델이 사용 가능한 상태인지 확인
        
        Returns:
            모델이 준비된 상태이면 True, 아니면 False
        """
        return self.model is not None and self.is_trained
    
    def get_config(self) -> Dict[str, Any]:
        """현재 모델 설정 반환
        
        Returns:
            모델 설정 딕셔너리
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """모델 설정 업데이트
        
        Args:
            new_config: 새로운 설정 딕셔너리
        """
        self.config.update(new_config)
        self.logger.info(f"설정 업데이트 완료: {new_config}")
    
    def __str__(self) -> str:
        """모델 정보 문자열 표현"""
        return f"{self.__class__.__name__}(trained={self.is_trained}, model_path={self.model_path})"
    
    def __repr__(self) -> str:
        """모델 정보 상세 표현"""
        return f"{self.__class__.__name__}(config={self.config}, trained={self.is_trained})" 