"""
EfficientNet 백본 모델 구현
"""

from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
)
from tensorflow.keras.applications.efficientnet import preprocess_input

from .base_backbone import BaseBackbone

class EfficientNetBackbone(BaseBackbone):
    """EfficientNet 백본 모델"""
    
    def __init__(self, weights: str = 'imagenet'):
        super().__init__(weights)
        self.model_map = {
            'EfficientNetB0': EfficientNetB0,
            'EfficientNetB1': EfficientNetB1,
            'EfficientNetB2': EfficientNetB2,
            'EfficientNetB3': EfficientNetB3,
            'EfficientNetB4': EfficientNetB4,
            'EfficientNetB5': EfficientNetB5,
            'EfficientNetB6': EfficientNetB6,
            'EfficientNetB7': EfficientNetB7
        }
        
    def build_backbone(self, inputs: tf.Tensor, model_name: str = 'EfficientNetB0', **kwargs) -> tf.Tensor:
        """EfficientNet 백본 모델 구축
        
        Args:
            inputs: 입력 텐서
            model_name: 사용할 EfficientNet 모델명
            **kwargs: 추가 파라미터
            
        Returns:
            EfficientNet의 출력 특성
        """
        if model_name not in self.model_map:
            raise ValueError(f"지원하지 않는 EfficientNet 모델: {model_name}")
        
        # 모델 생성
        model_class = self.model_map[model_name]
        self.backbone = model_class(
            weights=self.weights,
            include_top=False,
            input_tensor=inputs,
            **kwargs
        )
        
        return self.backbone.output
    
    def get_preprocessing_function(self):
        """EfficientNet 전처리 함수 반환"""
        return preprocess_input
    
    def get_input_shape(self, model_name: str) -> tuple:
        """모델별 입력 이미지 크기 반환
        
        Args:
            model_name: EfficientNet 모델명
            
        Returns:
            입력 이미지 크기 (height, width, channels)
        """
        shape_map = {
            'EfficientNetB0': (224, 224, 3),
            'EfficientNetB1': (240, 240, 3),
            'EfficientNetB2': (260, 260, 3),
            'EfficientNetB3': (300, 300, 3),
            'EfficientNetB4': (380, 380, 3),
            'EfficientNetB5': (456, 456, 3),
            'EfficientNetB6': (528, 528, 3),
            'EfficientNetB7': (600, 600, 3)
        }
        
        return shape_map.get(model_name, (224, 224, 3)) 