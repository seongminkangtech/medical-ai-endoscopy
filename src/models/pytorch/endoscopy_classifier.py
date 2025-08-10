"""
내시경 위치 분류를 위한 PyTorch 모델 클래스

이 모듈은 내시경 이미지에서 위치를 분류하는 다양한 CNN 모델을 제공합니다.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class EndoscopyClassifier(nn.Module):
    """
    내시경 위치 분류를 위한 PyTorch 모델 클래스
    
    다양한 pre-trained CNN 모델을 지원하며, 내시경 이미지 분류에 최적화되어 있습니다.
    """
    
    def __init__(self, 
                 model_name: str = "resnet50",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 freeze_backbone: bool = False):
        """
        모델 초기화
        
        Args:
            model_name: 사용할 모델 이름 (resnet50, inception_v3, efficientnet_b0 등)
            num_classes: 분류할 클래스 수
            pretrained: ImageNet pre-trained 가중치 사용 여부
            dropout_rate: Dropout 비율
            freeze_backbone: 백본 모델 가중치 고정 여부
        """
        super(EndoscopyClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        # 모델 백본 생성
        self.backbone = self._create_backbone(model_name, pretrained)
        
        # 분류 헤드 생성
        self.classifier = self._create_classifier(model_name, num_classes, dropout_rate)
        
        # 백본 가중치 고정 (선택사항)
        if freeze_backbone:
            self._freeze_backbone()
            
        logger.info(f"EndoscopyClassifier 초기화 완료: {model_name}, 클래스 수: {num_classes}")
    
    def _create_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """백본 모델 생성"""
        model_name = model_name.lower()
        
        if model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            # 마지막 FC 레이어 제거
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
            
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
            
        elif model_name == "inception_v3":
            model = models.inception_v3(pretrained=pretrained, aux_logits=False)
            # InceptionV3는 특별한 구조를 가짐
            self.feature_dim = 2048
            return model
            
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            # 마지막 분류 레이어 제거
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 1280
            
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 1024
            
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512
            
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        return model
    
    def _create_classifier(self, model_name: str, num_classes: int, dropout_rate: float) -> nn.Module:
        """분류 헤드 생성"""
        if model_name == "inception_v3":
            # InceptionV3는 특별한 구조
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        else:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
    
    def _freeze_backbone(self):
        """백본 모델 가중치 고정"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("백본 모델 가중치가 고정되었습니다.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 (batch_size, channels, height, width)
            
        Returns:
            분류 로짓 (batch_size, num_classes)
        """
        if self.model_name == "inception_v3":
            # InceptionV3는 특별한 처리 필요
            if self.training:
                x = self.backbone(x)
            else:
                x = self.backbone(x)
            x = self.classifier(x)
        else:
            x = self.backbone(x)
            x = self.classifier(x)
        
        return x
    
    def get_feature_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """
        특징 벡터 추출 (transfer learning용)
        
        Args:
            x: 입력 이미지 텐서
            
        Returns:
            특징 벡터 (batch_size, feature_dim)
        """
        if self.model_name == "inception_v3":
            x = self.backbone(x)
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = nn.Flatten()(x)
        else:
            x = self.backbone(x)
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = nn.Flatten()(x)
        
        return x
    
    def unfreeze_backbone(self):
        """백본 모델 가중치 해제 (fine-tuning용)"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("백본 모델 가중치가 해제되었습니다.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "feature_dimension": self.feature_dim
        } 