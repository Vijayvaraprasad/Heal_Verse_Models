from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adaptive_fitness_controller import AdaptiveFitnessController
from config import API_HOST, API_PORT, CHECKPOINT_DIR, DEVICE
from fusion.exercise_guidance import AbnormalityExerciseAdvisor

app = FastAPI(
    title="HealthFusionAI Model Server",
    version="3.0.0",
    description="Real-model inference API for Streamlit + fusion guidance",
)

# Evaluated best models
MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "brain_resnet": {
        "class": "brain_model.model_resnet:BrainResNet",
        "checkpoint": "ResNet_BEST.pt",
        "domain": "brain",
        "input_type": "image",
        "init_kwargs": {},
    },
    "bone_resnet18": {
        "class": "bone_model.bone_ai.model_classic:BoneResNet18",
        "checkpoint": "bone_model/bone_ai/checkpoints/best_classic.pth",
        "domain": "bone",
        "input_type": "image",
        "init_kwargs": {"pretrained": False, "freeze_backbone": False},
    },
    "cellular_risk": {
        "class": "cellular_model.cellular_ai.model_classic:ClassicalRiskModel",
        "checkpoint": "checkpoints/sweep_c_e50_lr8e4_wd1e4/best_classical.pt",
        "domain": "cellular",
        "input_type": "tabular",
        "init_kwargs": {"input_dim": 11},
    },
}

MODEL_PERFORMANCE = {
    "brain_resnet": {"accuracy": 0.9718579234972677, "macro_f1": 0.9610152214618981, "auc": 0.9916444413411925},
    "bone_resnet18": {"accuracy": 0.764200184218606, "macro_f1": 0.7604222168454128, "auc": 0.8314491636825168},
    "cellular_risk": {"accuracy": 0.9680555555555556, "macro_f1": 0.9625529143601432, "auc": 0.9948156773553599},
}

LOADED_MODELS: Dict[str, torch.nn.Module] = {}


class PredictionRequest(BaseModel):
    model: str
    data: Any
    shape: List[int]
    body_part_hint: Optional[str] = None
    condition_hint: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    model: str
    samples: Any
    shape: List[int]


class PredictionResponse(BaseModel):
    predictions: Union[List[float], List[List[float]]]
    batch_size: int
    model: str
    inference_mode: Literal["model"]
    predicted_label: Optional[Union[str, List[str]]] = None
    abnormal_probability: Optional[Union[float, List[float]]] = None
    abnormality_details: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None


class AbnormalityItem(BaseModel):
    domain: Literal["brain", "bone", "cellular", "other"] = "other"
    body_part: str
    condition: str = "abnormality"
    severity: float = 0.5


class ExerciseRecommendationRequest(BaseModel):
    abnormalities: List[AbnormalityItem]
    current_workout_intensity: Literal["low", "moderate", "high"] = "moderate"
    latest_biomarker_scores: Optional[Dict[str, float]] = None


class ExerciseRecommendationResponse(BaseModel):
    summary: Dict[str, Any]
    cautions: List[str]
    avoid_exercises: List[str]
    alternative_exercises: List[str]
    intensity_plan: Dict[str, Any]


def _resolve_model_class(spec: str):
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_checkpoint(path_str: str) -> Path:
    raw = Path(path_str)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path(CHECKPOINT_DIR) / raw)
        candidates.append(Path.cwd() / raw)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"checkpoint not found for '{path_str}'")


def _extract_state_dict(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return state
    for key in ("state_dict", "model_state", "model_state_dict"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def _try_load_model(model_name: str):
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]
    if model_name not in MODEL_INFO:
        raise KeyError(f"unknown model '{model_name}'")

    info = MODEL_INFO[model_name]
    cls = _resolve_model_class(info["class"])
    model = cls(**info.get("init_kwargs", {}))
    ckpt_path = _resolve_checkpoint(info["checkpoint"])

    state = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(_extract_state_dict(state))
    model.to(DEVICE)
    model.eval()
    LOADED_MODELS[model_name] = model
    return model


def _normalize_to_batch(data: Any, shape: List[int]) -> np.ndarray:
    if len(shape) == 0:
        raise ValueError("shape cannot be empty")
    arr = np.asarray(data, dtype=np.float32)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"shape mismatch: got {arr.size} values, expected {expected}")
    return arr.reshape(*shape)


def _normalize_output_row(row: np.ndarray) -> List[float]:
    vals = np.asarray(row, dtype=np.float32).reshape(-1)
    if vals.size == 1:
        p1 = 1.0 / (1.0 + float(np.exp(-vals[0])))
        return [round(1.0 - p1, 6), round(p1, 6)]
    shifted = vals - np.max(vals)
    exps = np.exp(shifted)
    probs = exps / max(float(np.sum(exps)), 1e-8)
    return [float(round(probs[0], 6)), float(round(probs[1], 6))]


def _prepare_tensor_for_model(model: torch.nn.Module, arr: np.ndarray) -> torch.Tensor:
    x = torch.tensor(arr, dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim == 4:
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        model_in_ch = None
        if hasattr(model, "model") and hasattr(model.model, "conv1"):
            model_in_ch = int(model.model.conv1.in_channels)
        elif hasattr(model, "conv1") and hasattr(model.conv1, "in_channels"):
            model_in_ch = int(model.conv1.in_channels)

        if model_in_ch is not None and x.shape[1] != model_in_ch:
            if x.shape[1] == 1 and model_in_ch == 3:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 3 and model_in_ch == 1:
                x = x.mean(dim=1, keepdim=True)

    return x


def _predict_scores(model_name: str, arr: np.ndarray) -> List[List[float]]:
    model = _try_load_model(model_name)
    x = _prepare_tensor_for_model(model, arr)
    with torch.no_grad():
        out = model(x.to(DEVICE))
    out_np = out.detach().cpu().numpy()
    if out_np.ndim == 1:
        out_np = out_np.reshape(1, -1)
    return [_normalize_output_row(row) for row in out_np]


def _prediction_meta(scores: List[float]) -> tuple[str, float]:
    abnormal_prob = float(scores[1] if len(scores) > 1 else scores[0])
    return ("abnormal" if abnormal_prob >= 0.5 else "normal"), abnormal_prob


def _build_abnormality(domain: str, scores: List[float], body_part: Optional[str], condition: Optional[str]) -> Dict[str, Any]:
    severity = float(scores[1] if len(scores) > 1 else scores[0])
    default_body_part = "brain" if domain == "brain" else ("systemic" if domain == "cellular" else "unknown")
    return {
        "domain": domain,
        "body_part": body_part or default_body_part,
        "condition": condition or "abnormality",
        "severity": round(max(0.0, min(1.0, severity)), 6),
    }


@app.get("/")
def root():
    return {"service": "HealthFusionAI Model Server", "version": app.version, "device": DEVICE, "total_models": len(MODEL_INFO)}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    output: Dict[str, Dict[str, Dict[str, str]]] = {"brain": {}, "bone": {}, "cellular": {}}
    for model_name, info in MODEL_INFO.items():
        output[info["domain"]][model_name] = {"checkpoint": info["checkpoint"], "input_type": info["input_type"]}
    return output


@app.get("/model_performance")
def model_performance():
    return MODEL_PERFORMANCE


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if req.model not in MODEL_INFO:
        raise HTTPException(status_code=404, detail=f"model '{req.model}' not found")

    try:
        batch = _normalize_to_batch(req.data, req.shape)
        scores = _predict_scores(req.model, batch)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"real-model inference failed: {e}")

    domain = MODEL_INFO[req.model]["domain"]
    abnormalities = [_build_abnormality(domain, s, req.body_part_hint, req.condition_hint) for s in scores]

    if len(scores) == 1:
        label, p = _prediction_meta(scores[0])
        return PredictionResponse(
            predictions=scores[0],
            batch_size=1,
            model=req.model,
            inference_mode="model",
            predicted_label=label,
            abnormal_probability=p,
            abnormality_details=abnormalities[0],
        )

    labels: List[str] = []
    probs: List[float] = []
    for s in scores:
        label, p = _prediction_meta(s)
        labels.append(label)
        probs.append(p)

    return PredictionResponse(
        predictions=scores,
        batch_size=len(scores),
        model=req.model,
        inference_mode="model",
        predicted_label=labels,
        abnormal_probability=probs,
        abnormality_details=abnormalities,
    )


@app.post("/predict_batch", response_model=PredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    return predict(PredictionRequest(model=req.model, data=req.samples, shape=req.shape))


@app.post("/recommend_exercises", response_model=ExerciseRecommendationResponse)
def recommend_exercises(req: ExerciseRecommendationRequest):
    guidance = AbnormalityExerciseAdvisor.build_guidance([a.model_dump() for a in req.abnormalities])
    controller = AdaptiveFitnessController()
    intensity_plan = controller.recommend(
        {
            "latest_biomarker_scores": req.latest_biomarker_scores or {},
            "current_workout_intensity": req.current_workout_intensity,
        }
    )
    return ExerciseRecommendationResponse(
        summary=guidance["summary"],
        cautions=guidance["cautions"],
        avoid_exercises=guidance["avoid_exercises"],
        alternative_exercises=guidance["alternative_exercises"],
        intensity_plan=intensity_plan,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
