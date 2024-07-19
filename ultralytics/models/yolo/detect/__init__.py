# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

from .train_m import DetectionTrainer_m
from .val_m import DetectionValidator_m

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "DetectionTrainer_m", "DetectionValidator_m"
