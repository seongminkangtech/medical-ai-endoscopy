"""
내시경 모델 학습을 위한 PyTorch Trainer 클래스

이 모듈은 내시경 분류 모델의 학습, 검증, 테스트를 관리합니다.
"""

import os
import time
import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm

from src.models.pytorch.endoscopy_classifier import EndoscopyClassifier
from src.utils.evaluation.model_evaluator import EndoscopyModelEvaluator

logger = logging.getLogger(__name__)

class EndoscopyTrainer:
    """
    내시경 모델 학습 클래스
    
    모델 학습, 검증, 테스트를 체계적으로 관리합니다.
    """
    
    def __init__(self, 
                 model: EndoscopyClassifier,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        트레이너 초기화
        
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            test_loader: 테스트 데이터 로더 (선택사항)
            config: 학습 설정
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # CUDA 최적화
        if torch.cuda.is_available():
            cudnn.benchmark = True
        
        # 학습 설정 초기화
        self._setup_training()
        
        # 평가기 초기화
        self.evaluator = EndoscopyModelEvaluator(
            class_names=self.config.get('class_names'),
            output_dir=self.config.get('output_dir', 'training_results')
        )
        
        # 학습 기록
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        logger.info(f"EndoscopyTrainer 초기화 완료 - 디바이스: {self.device}")
    
    def _setup_training(self):
        """학습 설정 초기화"""
        # 손실 함수
        if self.config.get('loss_function') == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.get('loss_function') == 'focal':
            self.criterion = self._focal_loss
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저
        if self.config.get('optimizer') == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        elif self.config.get('optimizer') == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3),
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3)
            )
        
        # 학습률 스케줄러
        if self.config.get('scheduler') == 'step':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif self.config.get('scheduler') == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif self.config.get('scheduler') == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('factor', 0.1),
                patience=self.config.get('patience', 10),
                verbose=True
            )
        else:
            self.scheduler = None
        
        logger.info(f"학습 설정 완료 - 옵티마이저: {type(self.optimizer).__name__}, "
                   f"학습률: {self.config.get('learning_rate', 1e-3)}")
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor, 
                    alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal Loss 구현"""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def train_epoch(self) -> Tuple[float, float]:
        """한 에포크 학습"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="학습")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 순전파
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 진행률 표시
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """한 에포크 검증"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="검증")
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 순전파
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 통계
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 진행률 표시
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, 
              epochs: int,
              save_dir: str = "checkpoints",
              save_best: bool = True,
              early_stopping: bool = True,
              patience: int = 15) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            epochs: 학습 에포크 수
            save_dir: 체크포인트 저장 디렉토리
            save_best: 최고 성능 모델 저장 여부
            early_stopping: 조기 종료 사용 여부
            patience: 조기 종료 인내심
            
        Returns:
            학습 결과 딕셔너리
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        logger.info(f"학습 시작 - 총 {epochs} 에포크")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc = self.validate_epoch()
            
            # 학습률 조정
            if self.scheduler is not None:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 학습 기록
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rate'].append(current_lr)
            
            # 에포크 정보 출력
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                       f"LR: {current_lr:.6f} - "
                       f"Time: {epoch_time:.2f}s")
            
            # 최고 성능 모델 저장
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                self.save_checkpoint(best_model_path, epoch, best_val_acc)
                logger.info(f"새로운 최고 성능 모델 저장: {best_model_path} (정확도: {best_val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 정기 체크포인트 저장
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path, epoch, val_acc)
                logger.info(f"체크포인트 저장: {checkpoint_path}")
            
            # 조기 종료
            if early_stopping and patience_counter >= patience:
                logger.info(f"조기 종료: {patience} 에포크 동안 성능 향상 없음")
                break
        
        # 최종 모델 저장
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        self.save_checkpoint(final_model_path, epochs-1, val_acc)
        
        # 학습 시간 계산
        total_time = time.time() - start_time
        logger.info(f"학습 완료 - 총 시간: {total_time/3600:.2f}시간")
        
        # 학습 결과 반환
        results = {
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'total_time': total_time,
            'epochs_trained': epoch + 1,
            'train_history': self.train_history
        }
        
        return results
    
    def test(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        모델 테스트
        
        Args:
            model_path: 테스트할 모델 경로 (None이면 현재 모델)
            
        Returns:
            테스트 결과 딕셔너리
        """
        if self.test_loader is None:
            logger.warning("테스트 데이터 로더가 설정되지 않았습니다.")
            return {}
        
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
            logger.info(f"모델 로드: {model_path}")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="테스트")
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 예측
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # 결과 수집
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 평가
        test_results = self.evaluator.evaluate_predictions(
            y_true=all_targets,
            y_pred=all_predictions,
            y_prob=np.array(all_probabilities)
        )
        
        logger.info(f"테스트 완료 - 정확도: {test_results['accuracy']:.4f}")
        
        return test_results
    
    def save_checkpoint(self, 
                       filepath: str,
                       epoch: int,
                       val_acc: float) -> None:
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'train_history': self.train_history,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"체크포인트 저장: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """체크포인트 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint.get('train_history', self.train_history)
        
        logger.info(f"체크포인트 로드: {filepath} (에포크: {checkpoint['epoch']}, "
                   f"검증 정확도: {checkpoint['val_acc']:.2f}%)")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """학습 히스토리 시각화"""
        if not self.train_history['train_loss']:
            logger.warning("학습 히스토리가 비어있습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 손실 그래프
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='학습 손실')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='검증 손실')
        axes[0, 0].set_title('학습 및 검증 손실')
        axes[0, 0].set_xlabel('에포크')
        axes[0, 0].set_ylabel('손실')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 정확도 그래프
        axes[0, 1].plot(epochs, self.train_history['train_acc'], 'b-', label='학습 정확도')
        axes[0, 1].plot(epochs, self.train_history['val_acc'], 'r-', label='검증 정확도')
        axes[0, 1].set_title('학습 및 검증 정확도')
        axes[0, 1].set_xlabel('에포크')
        axes[0, 1].set_ylabel('정확도 (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 학습률 그래프
        axes[1, 0].plot(epochs, self.train_history['learning_rate'], 'g-')
        axes[1, 0].set_title('학습률 변화')
        axes[1, 0].set_xlabel('에포크')
        axes[1, 0].set_ylabel('학습률')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 손실 vs 정확도
        axes[1, 1].scatter(self.train_history['train_loss'], self.train_history['train_acc'], 
                           alpha=0.6, label='학습')
        axes[1, 1].scatter(self.train_history['val_loss'], self.train_history['val_acc'], 
                           alpha=0.6, label='검증')
        axes[1, 1].set_title('손실 vs 정확도')
        axes[1, 1].set_xlabel('손실')
        axes[1, 1].set_ylabel('정확도 (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"학습 히스토리 저장: {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """모델 요약 정보 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'config': self.config,
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'test_samples': len(self.test_loader.dataset) if self.test_loader else 0
        } 