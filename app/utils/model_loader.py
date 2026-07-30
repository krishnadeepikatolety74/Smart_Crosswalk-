import os
import logging
import torch

# Monkeypatch torch.load to set weights_only=False by default for PyTorch 2.6 compatibility with YOLO
_orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO  # noqa: E402

_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
        
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    custom_model_path = os.path.join(base_dir, 'yolov10_custom.pt')
    default_model_path = os.path.join(base_dir, 'yolov10n.pt')
    
    # Try custom model (BDD100K + Cityscapes) first, fallback to yolov10n
    target_path = custom_model_path if os.path.exists(custom_model_path) else default_model_path
        
    try:
        _yolo_model = YOLO(target_path)
        logging.info(f"Successfully loaded YOLO model from {target_path}")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        _yolo_model = None
        
    return _yolo_model
