"""
위장 위치 분류를 위한 TensorFlow 모델 클래스

이 모듈은 다양한 백본 모델을 사용하여 위장 위치를 분류하는 모델을 제공합니다.
"""

from typing import Dict, Any, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os
from pathlib import Path

from ..base.base_model import BaseModel
from .backbones import (
    EfficientNetBackbone, ResNetBackbone, VGGBackbone, 
    DenseNetBackbone, InceptionBackbone, MobileNetBackbone
)

class GastricClassifier(BaseModel):
    """위장 위치 분류 모델
    
    다양한 백본 모델을 사용하여 위장 이미지의 위치를 분류합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """모델 초기화
        
        Args:
            config: 모델 설정 딕셔너리
        """
        super().__init__(config)
        
        # 백본 모델 설정
        self.backbone_name = config.get('backbone', 'EfficientNetB0')
        self.weights = config.get('weights', 'imagenet')
        self.input_shape = self._get_input_shape()
        
        # 백본 모델 생성
        self.backbone = self._create_backbone()
        
        # 분류기 설정
        self.classifier_type = config.get('classifier_type', 'gap')
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        # 훈련 설정
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.optimizer_name = config.get('optimizer', 'adam')
        
    def _get_input_shape(self) -> Tuple[int, int, int]:
        """백본 모델에 따른 입력 이미지 크기 반환"""
        # EfficientNet
        if 'efficient' in self.backbone_name.lower():
            backbone = EfficientNetBackbone()
            return backbone.get_input_shape(self.backbone_name)
        
        # ResNet
        elif 'resnet' in self.backbone_name.lower():
            return (224, 224, 3)
        
        # VGG
        elif 'vgg' in self.backbone_name.lower():
            return (224, 224, 3)
        
        # Inception
        elif 'inception' in self.backbone_name.lower():
            return (299, 299, 3)
        
        # MobileNet
        elif 'mobilenet' in self.backbone_name.lower():
            return (224, 224, 3)
        
        # 기본값
        else:
            return (224, 224, 3)
    
    def _create_backbone(self):
        """백본 모델 생성"""
        backbone_map = {
            'efficient': EfficientNetBackbone,
            'resnet': ResNetBackbone,
            'vgg': VGGBackbone,
            'densenet': DenseNetBackbone,
            'inception': InceptionBackbone,
            'mobilenet': MobileNetBackbone
        }
        
        # 백본 타입 결정
        for key, backbone_class in backbone_map.items():
            if key in self.backbone_name.lower():
                return backbone_class(weights=self.weights)
        
        raise ValueError(f"지원하지 않는 백본 모델: {self.backbone_name}")
    
    def build_model(self, input_shape: Optional[Tuple[int, ...]] = None, num_classes: int = 2) -> tf.keras.Model:
        """모델 아키텍처 구축
        
        Args:
            input_shape: 입력 데이터의 형태 (None이면 자동 설정)
            num_classes: 분류할 클래스 수
            
        Returns:
            구축된 모델 객체
        """
        if input_shape is None:
            input_shape = self.input_shape
        
        # 입력 레이어
        inputs = layers.Input(shape=input_shape)
        
        # 백본 모델 적용
        base_features = self.backbone.build_backbone(
            inputs, 
            model_name=self.backbone_name
        )
        
        # 분류기 구조
        x = self._build_classifier(base_features, num_classes)
        
        # 출력 레이어
        if num_classes == 2:
            outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs, outputs)
        
        # 모델 컴파일
        self._compile_model()
        
        return self.model
    
    def _build_classifier(self, features: tf.Tensor, num_classes: int) -> tf.Tensor:
        """분류기 구조 구축
        
        Args:
            features: 백본 모델의 특성 맵
            num_classes: 클래스 수
            
        Returns:
            분류를 위한 특성 벡터
        """
        if self.classifier_type == 'gap':
            return self._classifier_gap(features)
        elif self.classifier_type == 'f256x2':
            return self._classifier_f256x2(features)
        elif self.classifier_type == '512x2':
            return self._classifier_512x2(features)
        else:
            return self._classifier_gap(features)
    
    def _classifier_gap(self, x: tf.Tensor) -> tf.Tensor:
        """Global Average Pooling 기반 분류기"""
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        return x
    
    def _classifier_f256x2(self, x: tf.Tensor) -> tf.Tensor:
        """1024x1024 Dense 레이어 기반 분류기"""
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        return x
    
    def _classifier_512x2(self, x: tf.Tensor) -> tf.Tensor:
        """512x512 Dense 레이어 기반 분류기"""
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        return x
    
    def _compile_model(self):
        """모델 컴파일"""
        if self.model is None:
            raise ValueError("모델이 아직 구축되지 않았습니다.")
        
        # 옵티마이저 설정
        if self.optimizer_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # 손실 함수 설정
        if self.model.output_shape[-1] == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def train(self, train_data: Any, val_data: Any, **kwargs) -> Dict[str, Any]:
        """모델 훈련
        
        Args:
            train_data: 훈련 데이터
            val_data: 검증 데이터
            **kwargs: 추가 훈련 파라미터
            
        Returns:
            훈련 결과를 담은 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델을 먼저 구축해야 합니다.")
        
        # 훈련 파라미터 설정
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        callbacks = kwargs.get('callbacks', [])
        
        # 훈련 실행
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
    
    def predict(self, input_data: Any) -> Any:
        """모델 추론
        
        Args:
            input_data: 추론할 입력 데이터
            
        Returns:
            모델 예측 결과
        """
        if not self.is_model_ready():
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        return self.model.predict(input_data)
    
    def save_model(self, path: Union[str, Path]) -> None:
        """모델 저장
        
        Args:
            path: 모델을 저장할 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(path))
        self.model_path = str(path)
        self.logger.info(f"모델이 저장되었습니다: {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """모델 로드
        
        Args:
            path: 모델을 로드할 경로
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        
        self.model = tf.keras.models.load_model(str(path))
        self.model_path = str(path)
        self.is_trained = True
        self.logger.info(f"모델이 로드되었습니다: {path}")
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """백본 모델 레이어 고정/해제
        
        Args:
            freeze: True면 고정, False면 해제
        """
        if self.backbone.backbone is None:
            return
        
        for layer in self.backbone.backbone.layers:
            layer.trainable = not freeze
        
        self.logger.info(f"백본 모델 레이어가 {'고정' if freeze else '해제'}되었습니다.")
    
    def unfreeze_last_layers(self, num_layers: int = 10) -> None:
        """마지막 레이어들을 훈련 가능하게 설정
        
        Args:
            num_layers: 훈련 가능하게 할 레이어 수
        """
        if self.backbone.backbone is None:
            return
        
        total_layers = len(self.backbone.backbone.layers)
        for i, layer in enumerate(self.backbone.backbone.layers):
            if i >= total_layers - num_layers:
                layer.trainable = True
            else:
                layer.trainable = False
        
        self.logger.info(f"마지막 {num_layers}개 레이어가 훈련 가능하게 설정되었습니다.") 