"""
내시경 모델 평가 유틸리티

이 모듈은 내시경 분류 모델의 성능을 평가하고 시각화하는 기능을 제공합니다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class EndoscopyModelEvaluator:
    """
    내시경 모델 평가 클래스
    
    다양한 평가 지표와 시각화를 제공합니다.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None, output_dir: str = "evaluation_results"):
        """
        평가기 초기화
        
        Args:
            class_names: 클래스 이름 리스트
            output_dir: 결과 저장 디렉토리
        """
        self.class_names = class_names
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        logger.info(f"EndoscopyModelEvaluator 초기화: {len(class_names) if class_names else 'Unknown'}개 클래스")
    
    def evaluate_predictions(self, 
                           y_true: Union[List, np.ndarray, torch.Tensor],
                           y_pred: Union[List, np.ndarray, torch.Tensor],
                           y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        예측 결과 평가
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            y_prob: 예측 확률 (선택사항)
            
        Returns:
            평가 결과 딕셔너리
        """
        # 텐서를 numpy 배열로 변환
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)
        
        # 기본 지표 계산
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # 클래스별 지표
        if self.class_names:
            for i, class_name in enumerate(self.class_names):
                if i < len(np.unique(y_true)):
                    mask = (y_true == i)
                    if np.sum(mask) > 0:
                        results[f'precision_{class_name}'] = precision_score(
                            y_true[mask], y_pred[mask], average='binary', zero_division=0
                        )
                        results[f'recall_{class_name}'] = recall_score(
                            y_true[mask], y_pred[mask], average='binary', zero_division=0
                        )
                        results[f'f1_{class_name}'] = f1_score(
                            y_true[mask], y_pred[mask], average='binary', zero_division=0
                        )
        
        # 혼동 행렬
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # 분류 리포트
        if self.class_names:
            results['classification_report'] = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )
        else:
            results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC 및 PR 곡선 (이진 분류 또는 다중 분류)
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:
                # 이진 분류
                results.update(self._calculate_binary_metrics(y_true, y_prob))
            else:
                # 다중 분류
                results.update(self._calculate_multiclass_metrics(y_true, y_prob))
        
        logger.info(f"모델 평가 완료 - 정확도: {results['accuracy']:.4f}")
        return results
    
    def _to_numpy(self, data: Union[List, np.ndarray, torch.Tensor]) -> np.ndarray:
        """데이터를 numpy 배열로 변환"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"지원하지 않는 데이터 타입: {type(data)}")
    
    def _calculate_binary_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """이진 분류 지표 계산"""
        # ROC 곡선
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # PR 곡선
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'precision_curve': precision,
            'recall_curve': recall,
            'pr_thresholds': pr_thresholds
        }
    
    def _calculate_multiclass_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """다중 분류 지표 계산"""
        n_classes = y_prob.shape[1]
        
        # One-vs-Rest 방식으로 ROC 및 PR 곡선 계산
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        precision = dict()
        recall = dict()
        pr_auc = dict()
        
        for i in range(n_classes):
            # i번째 클래스에 대한 이진 분류
            y_true_binary = (y_true == i).astype(int)
            y_prob_binary = y_prob[:, i]
            
            # ROC 곡선
            fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_prob_binary)
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # PR 곡선
            precision[i], recall[i], _ = precision_recall_curve(y_true_binary, y_prob_binary)
            pr_auc[i] = average_precision_score(y_true_binary, y_prob_binary)
        
        # 마이크로 평균
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
        pr_auc["micro"] = average_precision_score(y_true_bin, y_prob, average="micro")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall
        }
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray,
                             save_path: Optional[str] = None,
                             normalize: bool = True) -> None:
        """
        혼동 행렬 시각화
        
        Args:
            confusion_matrix: 혼동 행렬
            save_path: 저장 경로 (선택사항)
            normalize: 정규화 여부
        """
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = "정규화된 혼동 행렬"
        else:
            cm = confusion_matrix
            title = "혼동 행렬"
        
        # 히트맵 생성
        if self.class_names:
            sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       cmap='Blues', cbar=True)
        else:
            sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd',
                       cmap='Blues', cbar=True)
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('예측 레이블', fontsize=12)
        plt.ylabel('실제 레이블', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"혼동 행렬 저장: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, 
                        results: Dict[str, Any],
                        save_path: Optional[str] = None) -> None:
        """
        ROC 곡선 시각화
        
        Args:
            results: evaluate_predictions 결과
            save_path: 저장 경로 (선택사항)
        """
        if 'roc_auc' not in results:
            logger.warning("ROC 곡선 데이터가 없습니다.")
            return
        
        plt.figure(figsize=(10, 8))
        
        if isinstance(results['roc_auc'], dict):
            # 다중 분류
            colors = plt.cm.Set1(np.linspace(0, 1, len(results['roc_auc'])))
            
            for i, (class_idx, auc_score) in enumerate(results['roc_auc'].items()):
                if class_idx == 'micro':
                    label = f'Micro-average (AUC = {auc_score:.3f})'
                    color = 'deeppink'
                    linestyle = '--'
                else:
                    class_name = self.class_names[int(class_idx)] if self.class_names else f'Class {class_idx}'
                    label = f'{class_name} (AUC = {auc_score:.3f})'
                    color = colors[i]
                    linestyle = '-'
                
                plt.plot(results['fpr'][class_idx], results['tpr'][class_idx],
                        color=color, linestyle=linestyle, linewidth=2, label=label)
        else:
            # 이진 분류
            plt.plot(results['fpr'], results['tpr'], color='blue', linewidth=2,
                    label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
        
        # 대각선
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC 곡선', fontsize=16, pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC 곡선 저장: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, 
                                   results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> None:
        """
        Precision-Recall 곡선 시각화
        
        Args:
            results: evaluate_predictions 결과
            save_path: 저장 경로 (선택사항)
        """
        if 'pr_auc' not in results:
            logger.warning("PR 곡선 데이터가 없습니다.")
            return
        
        plt.figure(figsize=(10, 8))
        
        if isinstance(results['pr_auc'], dict):
            # 다중 분류
            colors = plt.cm.Set1(np.linspace(0, 1, len(results['pr_auc'])))
            
            for i, (class_idx, auc_score) in enumerate(results['pr_auc'].items()):
                if class_idx == 'micro':
                    label = f'Micro-average (AP = {auc_score:.3f})'
                    color = 'deeppink'
                    linestyle = '--'
                else:
                    class_name = self.class_names[int(class_idx)] if self.class_names else f'Class {class_idx}'
                    label = f'{class_name} (AP = {auc_score:.3f})'
                    color = colors[i]
                    linestyle = '-'
                
                plt.plot(results['recall_curve'][class_idx], results['precision_curve'][class_idx],
                        color=color, linestyle=linestyle, linewidth=2, label=label)
        else:
            # 이진 분류
            plt.plot(results['recall_curve'], results['precision_curve'], color='blue', linewidth=2,
                    label=f'PR Curve (AP = {results["pr_auc"]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall 곡선', fontsize=16, pad=20)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR 곡선 저장: {save_path}")
        
        plt.show()
    
    def plot_class_distribution(self, 
                              y_true: Union[List, np.ndarray, torch.Tensor],
                              save_path: Optional[str] = None) -> None:
        """
        클래스 분포 시각화
        
        Args:
            y_true: 실제 레이블
            save_path: 저장 경로 (선택사항)
        """
        y_true = self._to_numpy(y_true)
        
        plt.figure(figsize=(10, 6))
        
        if self.class_names:
            # 클래스별 샘플 수 계산
            unique_labels, counts = np.unique(y_true, return_counts=True)
            class_counts = dict(zip([self.class_names[i] for i in unique_labels], counts))
            
            # 막대 그래프
            plt.bar(class_counts.keys(), class_counts.values(), color='skyblue', alpha=0.7)
            plt.xlabel('클래스', fontsize=12)
            plt.ylabel('샘플 수', fontsize=12)
            plt.title('클래스별 샘플 분포', fontsize=16, pad=20)
            
            # 값 표시
            for i, (class_name, count) in enumerate(class_counts.items()):
                plt.text(i, count + max(counts) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=11)
        else:
            # 숫자 레이블로 표시
            plt.hist(y_true, bins=len(np.unique(y_true)), alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('클래스 레이블', fontsize=12)
            plt.ylabel('샘플 수', fontsize=12)
            plt.title('클래스별 샘플 분포', fontsize=16, pad=20)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"클래스 분포 저장: {save_path}")
        
        plt.show()
    
    def save_evaluation_results(self, 
                              results: Dict[str, Any],
                              filename: str = "evaluation_results.txt") -> str:
        """
        평가 결과를 텍스트 파일로 저장
        
        Args:
            results: evaluate_predictions 결과
            filename: 저장할 파일명
            
        Returns:
            저장된 파일 경로
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("내시경 모델 평가 결과\n")
            f.write("=" * 60 + "\n\n")
            
            # 기본 지표
            f.write("기본 성능 지표:\n")
            f.write("-" * 30 + "\n")
            f.write(f"정확도 (Accuracy): {results['accuracy']:.4f}\n")
            f.write(f"정밀도 (Precision) - Macro: {results['precision_macro']:.4f}\n")
            f.write(f"정밀도 (Precision) - Weighted: {results['precision_weighted']:.4f}\n")
            f.write(f"재현율 (Recall) - Macro: {results['recall_macro']:.4f}\n")
            f.write(f"재현율 (Recall) - Weighted: {results['recall_weighted']:.4f}\n")
            f.write(f"F1 점수 - Macro: {results['f1_macro']:.4f}\n")
            f.write(f"F1 점수 - Weighted: {results['f1_weighted']:.4f}\n\n")
            
            # 클래스별 지표
            if self.class_names:
                f.write("클래스별 성능 지표:\n")
                f.write("-" * 30 + "\n")
                for class_name in self.class_names:
                    if f'precision_{class_name}' in results:
                        f.write(f"{class_name}:\n")
                        f.write(f"  - 정밀도: {results[f'precision_{class_name}']:.4f}\n")
                        f.write(f"  - 재현율: {results[f'recall_{class_name}']:.4f}\n")
                        f.write(f"  - F1 점수: {results[f'f1_{class_name}']:.4f}\n\n")
            
            # ROC AUC
            if 'roc_auc' in results:
                f.write("ROC AUC:\n")
                f.write("-" * 30 + "\n")
                if isinstance(results['roc_auc'], dict):
                    for class_idx, auc_score in results['roc_auc'].items():
                        if class_idx == 'micro':
                            f.write(f"Micro-average: {auc_score:.4f}\n")
                        else:
                            class_name = self.class_names[int(class_idx)] if self.class_names else f'Class {class_idx}'
                            f.write(f"{class_name}: {auc_score:.4f}\n")
                else:
                    f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
                f.write("\n")
            
            # 혼동 행렬
            f.write("혼동 행렬:\n")
            f.write("-" * 30 + "\n")
            if self.class_names:
                f.write("실제/예측\t" + "\t".join(self.class_names) + "\n")
                for i, row in enumerate(results['confusion_matrix']):
                    f.write(f"{self.class_names[i]}\t" + "\t".join(map(str, row)) + "\n")
            else:
                f.write(str(results['confusion_matrix']))
        
        logger.info(f"평가 결과 저장: {filepath}")
        return filepath 