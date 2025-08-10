"""
백본 모델의 기본 인터페이스 정의
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import tensorflow as tf
from tensorflow import keras

class BaseBackbone(ABC):
    """백본 모델의 기본 클래스"""
    
    def __init__(self, weights: str = 'imagenet'):
        """백본 모델 초기화
        
        Args:
            weights: 사용할 가중치 ('imagenet', 'random', None)
        """
        self.weights = weights
        self.backbone = None
        
    @abstractmethod
    def build_backbone(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """백본 모델 구축
        
        Args:
            inputs: 입력 텐서
            **kwargs: 추가 파라미터
            
        Returns:
            백본 모델의 출력 특성
        """
        pass
    
    @abstractmethod
    def get_preprocessing_function(self):
        """전처리 함수 반환
        
        Returns:
            이미지 전처리 함수
        """
        pass
    
    def freeze_layers(self, num_layers: int = 0) -> None:
        """특정 레이어들을 고정
        
        Args:
            num_layers: 고정할 레이어 수 (0이면 모든 레이어 고정)
        """
        if self.backbone is None:
            return
            
        if num_layers == 0:
            # 모든 레이어 고정
            for layer in self.backbone.layers:
                layer.trainable = False
        else:
            # 마지막 num_layers만 훈련 가능하게 설정
            total_layers = len(self.backbone.layers)
            for i, layer in enumerate(self.backbone.layers):
                if i < total_layers - num_layers:
                    layer.trainable = False
                else:
                    layer.trainable = True
    
    def get_feature_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """특성 맵의 출력 형태 반환
        
        Args:
            input_shape: 입력 이미지 형태
            
        Returns:
            특성 맵의 형태
        """
        if self.backbone is None:
            raise ValueError("백본 모델이 아직 구축되지 않았습니다.")
        
        # 더미 입력으로 특성 맵 형태 확인
        dummy_input = tf.random.normal((1,) + input_shape)
        features = self.backbone(dummy_input, training=False)
        return features.shape[1:] 