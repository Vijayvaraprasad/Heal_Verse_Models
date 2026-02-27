"""
Configuration and settings for HealthFusionAI API and frontend.
Centralize model paths, preprocessing defaults, and runtime parameters.
"""

import os
from pathlib import Path
from typing import Dict, Literal

# ===========================
# PATHS
# ===========================

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Model folders
BRAIN_MODEL_DIR = PROJECT_ROOT / "brain_model"
BONE_MODEL_DIR = PROJECT_ROOT / "bone_model"
CELLULAR_MODEL_DIR = PROJECT_ROOT / "cellular_model"

# Ensure paths exist
CHECKPOINT_DIR.mkdir(exist_ok=True)
for p in [BRAIN_MODEL_DIR, BONE_MODEL_DIR, CELLULAR_MODEL_DIR]:
    p.mkdir(exist_ok=True)


# ===========================
# IMAGE PREPROCESSING
# ===========================

IMAGE_SIZE: int = 64  # target size for all image models (H, W)
NORMALIZE_IMAGES: bool = True  # scale to [0,1]
IMAGE_CHANNELS: Literal[1, 3] = 1  # grayscale


# ===========================
# TABULAR PREPROCESSING
# ===========================

TABULAR_NORMALIZE: bool = True  # z-score normalization
TABULAR_HANDLE_MISSING: bool = True  # fill NaN/inf
MISSING_VALUE_STRATEGY: Literal["mean", "median", "zero"] = "mean"

# Default feature counts for each domain
FEATURE_DIMS = {
    "cellular": 20,  # adjust based on actual data
}


# ===========================
# MODEL CONFIGURATION
# ===========================

# For all classification models we assume binary output unless overridden
NUM_CLASSES: int = 2


# Brain models (best-performing selection only)
BRAIN_MODELS = {
    "resnet": {
        "checkpoint": "ResNet_BEST.pt",
        "init_kwargs": {},
    },
    "qml": {
        "checkpoint": "CNN_+_QML_BEST.pt",
        "init_kwargs": {},
    },
}

# Bone models (best available working checkpoint)
BONE_MODELS = {
    "resnet18": {
        "checkpoint": "bone_model/bone_ai/checkpoints/best_classic.pth",
        "init_kwargs": {"pretrained": False, "freeze_backbone": False},
    },
}

# Cellular models (best-performing classical checkpoint)
CELLULAR_MODELS = {
    "risk": {
        "checkpoint": "checkpoints/sweep_c_e50_lr8e4_wd1e4/best_classical.pt",
        "init_kwargs": {"input_dim": 11},
    },
}

# Model availability
ENABLE_BRAIN_MODELS: bool = True
ENABLE_BONE_MODELS: bool = True
ENABLE_CELLULAR_MODELS: bool = True


# ===========================
# API SETTINGS
# ===========================

API_HOST: str = "127.0.0.1"
API_PORT: int = 8000
API_RELOAD: bool = True  # set to False in production

# Request timeouts
REQUEST_TIMEOUT_SECONDS: int = 30

# Batch processing
MAX_BATCH_SIZE: int = 64  # max samples per request
DEFAULT_BATCH_SIZE: int = 1


# ===========================
# STREAMLIT SETTINGS
# ===========================

STREAMLIT_PAGE_TITLE: str = "HealthFusionAI Model Tester"
STREAMLIT_LAYOUT: Literal["centered", "wide"] = "wide"
DEFAULT_API_URL: str = f"http://{API_HOST}:{API_PORT}"

# UI defaults
DEFAULT_IMAGE_SIZE_DISPLAY: int = 200  # pixels in UI
SHOW_CONFIDENCE_CHART: bool = True
SHOW_RAW_PREDICTIONS: bool = True


# ===========================
# DEVICE & HARDWARE
# ===========================

import torch

USE_GPU: bool = torch.cuda.is_available()
DEVICE: str = "cuda" if USE_GPU else "cpu"
DTYPE: torch.dtype = torch.float32


# ===========================
# LOGGING
# ===========================

LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
LOG_FILE: str = str(PROJECT_ROOT / "logs" / "api.log")

# Create log dir
Path(LOG_FILE).parent.mkdir(exist_ok=True)


# ===========================
# RUNTIME HELPERS
# ===========================

def get_model_checkpoint_path(domain: str, model_name: str) -> Path:
    """Build full path to model checkpoint."""
    if domain == "brain":
        rel_path = BRAIN_MODELS.get(model_name, {}).get("checkpoint", "")
    elif domain == "bone":
        rel_path = BONE_MODELS.get(model_name, {}).get("checkpoint", "")
    elif domain == "cellular":
        rel_path = CELLULAR_MODELS.get(model_name, {}).get("checkpoint", "")
    else:
        return None
    
    return CHECKPOINT_DIR / rel_path


def get_model_init_kwargs(domain: str, model_name: str) -> Dict:
    """Get initialization kwargs for a model."""
    if domain == "brain":
        return BRAIN_MODELS.get(model_name, {}).get("init_kwargs", {})
    elif domain == "bone":
        return BONE_MODELS.get(model_name, {}).get("init_kwargs", {})
    elif domain == "cellular":
        return CELLULAR_MODELS.get(model_name, {}).get("init_kwargs", {})
    return {}


def get_available_models() -> Dict[str, Dict[str, Dict]]:
    """Return all available models organized by domain."""
    available = {}
    
    if ENABLE_BRAIN_MODELS:
        available["brain"] = {
            name: {"checkpoint": info["checkpoint"], "input_type": "image"}
            for name, info in BRAIN_MODELS.items()
        }
    
    if ENABLE_BONE_MODELS:
        available["bone"] = {
            name: {"checkpoint": info["checkpoint"], "input_type": "image"}
            for name, info in BONE_MODELS.items()
        }
    
    if ENABLE_CELLULAR_MODELS:
        available["cellular"] = {
            name: {"checkpoint": info["checkpoint"], "input_type": "tabular"}
            for name, info in CELLULAR_MODELS.items()
        }
    
    return available


# ===========================
# DEBUG
# ===========================

if __name__ == "__main__":
    print("HealthFusionAI Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"GPU Available: {USE_GPU}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print()
    print("Available Models:")
    for domain, models in get_available_models().items():
        print(f"  {domain.upper()}:")
        for name in models.keys():
            print(f"    - {name}")
