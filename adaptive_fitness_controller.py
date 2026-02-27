from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ControllerConfig:
    crp_high: float = 5.0
    alt_high: float = 56.0
    ast_high: float = 40.0
    inflammation_high: float = 0.70
    tumor_high: float = 0.70
    fatigue_high: float = 0.70
    liver_stress_high: float = 0.70


class LearnedLoadScorer(nn.Module):
    """Small learned component to estimate training tolerance (0..1)."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveFitnessController:
    """Hybrid adaptive workout controller (rule-based + learned)."""

    INTENSITY_ORDER = ["low", "moderate", "high"]

    def __init__(self, config: ControllerConfig | None = None, model: nn.Module | None = None) -> None:
        self.config = config or ControllerConfig()
        self.model = model or LearnedLoadScorer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _to_level(value: str) -> int:
        v = value.strip().lower()
        if v not in AdaptiveFitnessController.INTENSITY_ORDER:
            raise ValueError("current_workout_intensity must be one of: low, moderate, high")
        return AdaptiveFitnessController.INTENSITY_ORDER.index(v)

    @staticmethod
    def _from_level(level: int) -> str:
        level = max(0, min(2, level))
        return AdaptiveFitnessController.INTENSITY_ORDER[level]

    def _learned_tolerance(self, inputs: dict[str, float]) -> float:
        features = torch.tensor(
            [
                inputs.get("inflammation_score", 0.0),
                inputs.get("tumor_severity", 0.0),
                inputs.get("liver_stress_score", 0.0),
                inputs.get("fatigue_score", 0.0),
                inputs.get("CRP", 0.0) / 20.0,
                (inputs.get("ALT", 0.0) + inputs.get("AST", 0.0)) / 200.0,
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            tolerance = float(self.model(features).item())
        return max(0.0, min(1.0, tolerance))

    def recommend(self, payload: dict[str, Any]) -> dict[str, Any]:
        biomarkers = payload.get("latest_biomarker_scores", {})
        current_intensity = str(payload.get("current_workout_intensity", "moderate"))

        crp = float(biomarkers.get("CRP", 0.0))
        alt = float(biomarkers.get("ALT", 0.0))
        ast = float(biomarkers.get("AST", 0.0))
        inflammation = float(biomarkers.get("inflammation_score", 0.0))
        tumor = float(biomarkers.get("tumor_severity", 0.0))
        fatigue = float(biomarkers.get("fatigue_score", 0.0))
        liver_stress = float(biomarkers.get("liver_stress_score", 0.0))

        current_level = self._to_level(current_intensity)
        notes: list[str] = []
        overrides: list[str] = []

        # Learned signal: higher tolerance permits higher training intensity.
        tolerance = self._learned_tolerance(
            {
                "inflammation_score": inflammation,
                "tumor_severity": tumor,
                "liver_stress_score": liver_stress,
                "fatigue_score": fatigue,
                "CRP": crp,
                "ALT": alt,
                "AST": ast,
            }
        )

        if tolerance < 0.35:
            target_level = max(0, current_level - 1)
            notes.append("Learned model suggests reduced training tolerance.")
        elif tolerance > 0.70:
            target_level = min(2, current_level + 1)
            notes.append("Learned model suggests good training tolerance.")
        else:
            target_level = current_level

        # Rule-based safety overrides (always win).
        if crp > self.config.crp_high or inflammation > self.config.inflammation_high:
            target_level = 0
            overrides.append("CRP/inflammation high: intensity reduced for safety.")

        if alt > self.config.alt_high or ast > self.config.ast_high or liver_stress > self.config.liver_stress_high:
            target_level = 0
            overrides.append("ALT/AST or liver stress high: avoid heavy training.")

        if fatigue > self.config.fatigue_high:
            target_level = min(target_level, 0)
            overrides.append("Fatigue high: prioritize rest and recovery.")

        if tumor > self.config.tumor_high:
            target_level = min(target_level, 1)
            overrides.append("Tumor severity elevated: cap at moderate intensity.")

        adjusted_intensity = self._from_level(target_level)

        # Calorie adjustment relative to maintenance.
        if adjusted_intensity == "low":
            calorie_adjustment = -250
            rest_recommendation = "48h active recovery (walk, mobility, hydration)."
        elif adjusted_intensity == "moderate":
            calorie_adjustment = 0
            rest_recommendation = "24h recovery between intense sessions."
        else:
            calorie_adjustment = 200
            rest_recommendation = "Standard sleep/recovery protocol (7-9h sleep)."

        return {
            "status": "ok",
            "input_summary": {
                "current_workout_intensity": current_intensity.lower(),
                "biomarkers_used": {
                    "CRP": crp,
                    "ALT": alt,
                    "AST": ast,
                    "inflammation_score": inflammation,
                    "tumor_severity": tumor,
                    "liver_stress_score": liver_stress,
                    "fatigue_score": fatigue,
                },
            },
            "output": {
                "adjusted_workout_intensity": adjusted_intensity,
                "calorie_adjustment_kcal": calorie_adjustment,
                "rest_recommendation": rest_recommendation,
            },
            "decision_trace": {
                "learned_tolerance_score": round(tolerance, 4),
                "hybrid_notes": notes,
                "safety_overrides": overrides,
            },
        }


if __name__ == "__main__":
    controller = AdaptiveFitnessController()
    demo_input = {
        "latest_biomarker_scores": {
            "CRP": 12.0,
            "ALT": 62.0,
            "AST": 44.0,
            "inflammation_score": 0.82,
            "tumor_severity": 0.55,
            "liver_stress_score": 0.73,
            "fatigue_score": 0.78,
        },
        "current_workout_intensity": "high",
    }

    result = controller.recommend(demo_input)
    import json

    print(json.dumps(result, indent=2))
