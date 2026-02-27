from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExerciseRule:
    cautions: list[str]
    avoid: list[str]
    alternatives: list[str]


class AbnormalityExerciseAdvisor:
    """Maps abnormal body regions to practical training restrictions."""

    _RULES: dict[str, ExerciseRule] = {
        "finger": ExerciseRule(
            cautions=[
                "Limit strong gripping and closed-fist loading.",
                "Use pain-free range only.",
            ],
            avoid=["push-ups", "heavy barbell bench press", "heavy deadlift without straps"],
            alternatives=["machine chest press", "pec-deck fly", "cable chest fly"],
        ),
        "toe": ExerciseRule(
            cautions=[
                "Avoid forefoot impact and explosive push-off.",
                "Keep steps short and controlled.",
            ],
            avoid=["sprinting", "jump rope", "box jumps"],
            alternatives=["seated cycling", "rowing machine", "seated upper-body strength work"],
        ),
        "wrist": ExerciseRule(
            cautions=["Keep wrist neutral and avoid extension under load."],
            avoid=["floor push-ups", "front squat with extended wrist", "handstand drills"],
            alternatives=["dumbbell neutral-grip press", "machine press", "landmine press"],
        ),
        "hand": ExerciseRule(
            cautions=["Use straps or assisted grips if needed.", "Avoid crushing grip training."],
            avoid=["farmer carries", "heavy pull-ups", "thick-grip lifting"],
            alternatives=["machine rows", "lat pulldown with straps", "cable back work"],
        ),
        "elbow": ExerciseRule(
            cautions=["Reduce locking out under heavy load."],
            avoid=["heavy skull crushers", "heavy dips", "max-effort curls"],
            alternatives=["rope pressdowns", "light cable curls", "tempo-based machine work"],
        ),
        "shoulder": ExerciseRule(
            cautions=["Avoid painful overhead range and sudden jerks."],
            avoid=["behind-neck press", "upright row", "heavy overhead press"],
            alternatives=["landmine press", "incline machine press", "cable lateral raise"],
        ),
        "knee": ExerciseRule(
            cautions=["Avoid deep flexion if painful and limit impact."],
            avoid=["jump squats", "plyometric lunges", "hard downhill running"],
            alternatives=["leg press partial range", "hamstring curls", "cycling"],
        ),
        "ankle": ExerciseRule(
            cautions=["Prioritize stability and low-impact locomotion."],
            avoid=["cutting drills", "jumping drills", "trail running"],
            alternatives=["stationary bike", "seated calf work", "controlled balance drills"],
        ),
        "hip": ExerciseRule(
            cautions=["Limit deep loaded hip flexion if symptomatic."],
            avoid=["deep heavy squats", "ballistic kettlebell swings", "max sprints"],
            alternatives=["glute bridge", "hip thrust light-moderate load", "reverse sled pull"],
        ),
        "spine": ExerciseRule(
            cautions=["Keep neutral spine and avoid repeated end-range flexion."],
            avoid=["heavy axial loading", "good mornings", "high-volume sit-ups"],
            alternatives=["bird-dog", "dead-bug", "chest-supported rows"],
        ),
        "neck": ExerciseRule(
            cautions=["Avoid sudden neck loading and high-impact contact."],
            avoid=["heavy shrugs with poor form", "wrestling/impact drills", "neck bridges"],
            alternatives=["isometric neck rehab drills", "light cable rows", "walking"],
        ),
        "brain": ExerciseRule(
            cautions=["Avoid activities with fall risk while symptomatic."],
            avoid=["contact sports", "high-velocity head movement drills", "max-intensity intervals"],
            alternatives=["walking", "light cycling", "mobility and breathing sessions"],
        ),
    }

    _ALIASES: dict[str, str] = {
        "thumb": "finger",
        "index": "finger",
        "middle finger": "finger",
        "ring finger": "finger",
        "little finger": "finger",
        "feet": "toe",
        "foot": "toe",
        "shoulders": "shoulder",
        "knees": "knee",
        "ankles": "ankle",
        "back": "spine",
        "lower back": "spine",
    }

    @classmethod
    def _normalize_part(cls, part: str) -> str:
        key = str(part).strip().lower()
        return cls._ALIASES.get(key, key)

    @classmethod
    def _default_rule(cls) -> ExerciseRule:
        return ExerciseRule(
            cautions=["Use low-impact training and stop any movement that causes pain."],
            avoid=["max-effort lifting", "high-impact plyometrics"],
            alternatives=["machine-guided strength work", "walking", "mobility exercises"],
        )

    @classmethod
    def build_guidance(cls, abnormalities: list[dict[str, Any]]) -> dict[str, Any]:
        if not abnormalities:
            return {
                "summary": {
                    "status": "no_abnormality_reported",
                    "focus_parts": [],
                    "detected_abnormalities": [],
                },
                "cautions": [],
                "avoid_exercises": [],
                "alternative_exercises": ["full routine allowed if asymptomatic"],
            }

        cautions: set[str] = set()
        avoid: set[str] = set()
        alternatives: set[str] = set()
        focus_parts: set[str] = set()
        detected: list[dict[str, Any]] = []

        for item in abnormalities:
            part = cls._normalize_part(str(item.get("body_part", "unknown")))
            rule = cls._RULES.get(part, cls._default_rule())
            focus_parts.add(part)
            cautions.update(rule.cautions)
            avoid.update(rule.avoid)
            alternatives.update(rule.alternatives)
            detected.append(
                {
                    "domain": item.get("domain", "unknown"),
                    "body_part": part,
                    "condition": item.get("condition", "abnormality"),
                    "severity": float(item.get("severity", 0.5)),
                }
            )

        return {
            "summary": {
                "status": "abnormality_detected",
                "focus_parts": sorted(focus_parts),
                "detected_abnormalities": detected,
            },
            "cautions": sorted(cautions),
            "avoid_exercises": sorted(avoid),
            "alternative_exercises": sorted(alternatives),
        }
