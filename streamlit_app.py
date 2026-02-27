import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import streamlit as st
from PIL import Image
from pathlib import Path

try:
    from brain_model.ocr_report_parser import build_output, ocr_pages_from_pdf, ocr_single_image, parse_pages, parse_report
except Exception:
    build_output = None
    ocr_pages_from_pdf = None
    ocr_single_image = None
    parse_pages = None
    parse_report = None

try:
    from cellular_multilabel_model import predict_cellular
    from disease_registry import DISEASE_REGISTRY, due_date_from_urgency
    from feature_mapper import map_accepted_records_to_features
except Exception:
    predict_cellular = None
    DISEASE_REGISTRY = {}
    due_date_from_urgency = None
    map_accepted_records_to_features = None

try:
    import pytesseract  # type: ignore
    pytesseract.pytesseract.tesseract_cmd = os.getenv(
        "TESSERACT_CMD",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
except Exception:
    pytesseract = None

st.set_page_config(page_title="HealthFusionAI", layout="wide")

DEFAULT_API_URL = os.getenv("HF_API_URL", "http://127.0.0.1:18080")
API_URL = st.sidebar.text_input("API URL", DEFAULT_API_URL)
INTENSITY = st.sidebar.selectbox("Workout intensity", ["low", "moderate", "high"], index=1)

st.title("HealthFusionAI Tester")
st.caption("Rewritten stable frontend for backend endpoints: /models, /predict, /predict_and_recommend")


def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=20)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: Dict[str, Any]):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_scores(pred):
    if isinstance(pred, list) and pred and isinstance(pred[0], list):
        return pred[0]
    return pred


CANONICAL_14_FEATURES = [
    "wbc",
    "rbc",
    "hemoglobin",
    "platelets",
    "glucose",
    "cholesterol",
    "alt",
    "ast",
    "crp",
    "bilirubin",
    "albumin",
    "creatinine",
    "urea",
    "triglycerides",
]

# Current cellular model expects 11 values. Keep a deterministic subset mapping.
MODEL_11_FEATURES = [
    "wbc",
    "rbc",
    "glucose",
    "hemoglobin",
    "platelets",
    "cholesterol",
    "alt",
    "ast",
    "crp",
    "bilirubin",
    "creatinine",
]

LAB_ALIASES = {
    "wbc": [r"\bwbc\b", r"white\s*blood\s*cell"],
    "rbc": [r"\brbc\b", r"red\s*blood\s*cell"],
    "hemoglobin": [r"\bhb\b", r"haemoglobin", r"hemoglobin"],
    "platelets": [r"platelet", r"\bplt\b"],
    "glucose": [r"glucose", r"blood\s*sugar", r"\bfbs\b", r"\brbs\b"],
    "cholesterol": [r"cholesterol", r"total\s*chol"],
    "alt": [r"\balt\b", r"sgpt"],
    "ast": [r"\bast\b", r"sgot"],
    "crp": [r"\bcrp\b", r"c[-\s]*reactive\s*protein"],
    "bilirubin": [r"bilirubin"],
    "albumin": [r"albumin"],
    "creatinine": [r"creatinine"],
    "urea": [r"\burea\b", r"blood\s*urea"],
    "triglycerides": [r"triglyceride", r"\btg\b"],
}

REFERENCE_DEFAULTS = {
    "wbc": 7.0,
    "rbc": 4.7,
    "hemoglobin": 14.0,
    "platelets": 275.0,
    "glucose": 100.0,
    "cholesterol": 180.0,
    "alt": 25.0,
    "ast": 25.0,
    "crp": 1.0,
    "bilirubin": 0.8,
    "albumin": 4.2,
    "creatinine": 0.9,
    "urea": 30.0,
    "triglycerides": 140.0,
}

OCR_TO_CANONICAL = {
    "total_leucocyte_count": "wbc",
    "rbc_count": "rbc",
    "haemoglobin": "hemoglobin",
    "hemoglobin": "hemoglobin",
    "platelet_count": "platelets",
    "glucose": "glucose",
    "cholesterol": "cholesterol",
    "alt": "alt",
    "ast": "ast",
    "crp": "crp",
    "bilirubin": "bilirubin",
    "albumin": "albumin",
    "creatinine": "creatinine",
    "urea": "urea",
    "triglycerides": "triglycerides",
}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _derived_scores(features: Dict[str, float]) -> Dict[str, float]:
    crp = features["crp"]
    wbc = features["wbc"]
    alt = features["alt"]
    ast = features["ast"]
    bilirubin = features["bilirubin"]
    hemoglobin = features["hemoglobin"]
    rbc = features["rbc"]
    glucose = features["glucose"]

    if crp is not None:
        inflammation_score = _clip((crp - 1.0) / (10.0 - 1.0))
    else:
        inflammation_score = _clip((wbc - 7.0) / (15.0 - 7.0))

    alt_n = _clip((alt - 40.0) / (200.0 - 40.0))
    ast_n = _clip((ast - 40.0) / (200.0 - 40.0))
    bili_n = _clip((bilirubin - 1.2) / (3.0 - 1.2))
    liver_stress_score = _clip(0.45 * alt_n + 0.35 * ast_n + 0.20 * bili_n)

    hb_low = _clip((13.5 - hemoglobin) / (13.5 - 8.0))
    rbc_low = _clip((4.5 - rbc) / (4.5 - 3.0))
    glucose_var = _clip(abs(glucose - 100.0) / 80.0)
    fatigue_score = _clip(0.55 * hb_low + 0.25 * rbc_low + 0.20 * glucose_var)

    return {
        "inflammation_score": round(inflammation_score, 4),
        "liver_stress_score": round(liver_stress_score, 4),
        "fatigue_score": round(fatigue_score, 4),
    }


def fill_missing_features_with_formulas(features14: Dict[str, Optional[float]]) -> tuple[Dict[str, float], List[str]]:
    f: Dict[str, float] = {}
    imputed: List[str] = []
    for k in CANONICAL_14_FEATURES:
        v = features14.get(k)
        if v is None:
            f[k] = float(REFERENCE_DEFAULTS[k])
            imputed.append(k)
        else:
            f[k] = float(v)

    # Formula-based second pass for clinically-linked gaps
    if features14.get("crp") is None:
        f["crp"] = max(0.3, min(10.0, 1.0 + 0.25 * (f["wbc"] - 7.0)))
    if features14.get("bilirubin") is None:
        f["bilirubin"] = max(0.3, min(3.0, 0.6 + 0.004 * max(f["alt"] + f["ast"] - 50.0, 0.0)))
    if features14.get("creatinine") is None:
        f["creatinine"] = max(0.5, min(2.5, 0.6 + 0.015 * max(f["urea"] - 20.0, 0.0)))
    if features14.get("urea") is None:
        f["urea"] = max(12.0, min(80.0, 18.0 + 15.0 * max(f["creatinine"] - 0.8, 0.0)))
    if features14.get("triglycerides") is None:
        f["triglycerides"] = max(60.0, min(400.0, 0.75 * f["cholesterol"]))
    if features14.get("albumin") is None:
        scores = _derived_scores(f)
        f["albumin"] = max(2.8, min(5.4, 4.5 - 1.2 * scores["inflammation_score"]))

    return f, imputed


def _extract_text_from_report(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.read()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(uploaded_file)
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            st.warning(f"PDF parsing unavailable/failed ({e}). Paste report text manually below.")
            return ""
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        try:
            if pytesseract is None:
                raise RuntimeError("pytesseract package not available")
            img = Image.open(uploaded_file)
            return pytesseract.image_to_string(img)
        except Exception as e:
            st.warning(f"OCR unavailable/failed ({e}). Paste report text manually below.")
            return ""
    return ""


def _extract_first_number_near(text: str, token_pattern: str) -> Optional[float]:
    # Find number near marker within same line or short window.
    pattern = re.compile(token_pattern + r".{0,40}?([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_14_features_from_text(report_text: str) -> Dict[str, Optional[float]]:
    compact = " ".join(report_text.split())
    out: Dict[str, Optional[float]] = {k: None for k in CANONICAL_14_FEATURES}
    for feature, aliases in LAB_ALIASES.items():
        val: Optional[float] = None
        for alias in aliases:
            val = _extract_first_number_near(compact, alias)
            if val is not None:
                break
        out[feature] = val
    return out


def _parse_float_safe(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def extract_14_features_from_ocr_json(ocr_output: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in CANONICAL_14_FEATURES}
    accepted = ocr_output.get("accepted_records", [])
    if not isinstance(accepted, list):
        return out

    for rec in accepted:
        if not isinstance(rec, dict):
            continue
        test_name = str(rec.get("test_name", "")).strip().lower()
        mapped = OCR_TO_CANONICAL.get(test_name)
        if not mapped:
            continue
        val = _parse_float_safe(rec.get("value"))
        if val is None:
            continue
        out[mapped] = val
    return out


def run_ocr_pipeline_from_upload(uploaded_file) -> Optional[Dict[str, Any]]:
    if not all([build_output, ocr_pages_from_pdf, ocr_single_image, parse_pages]):
        st.warning("OCR parser module not available in this environment.")
        return None

    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return None

    data = uploaded_file.getvalue()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(data)
            tmp_path = tf.name

        if suffix == ".pdf":
            pages = ocr_pages_from_pdf(Path(tmp_path), dpi=300, lang="eng")
        else:
            pages = ocr_single_image(Path(tmp_path), lang="eng")

        rows = parse_pages(pages)
        out = build_output(pages, rows)
        return out
    except Exception as e:
        st.warning(f"OCR pipeline failed ({e}). Falling back to text extraction.")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def build_model_11_vector(features14: Dict[str, Optional[float]]) -> tuple[List[float], List[str]]:
    missing: List[str] = []
    vec: List[float] = []
    for name in MODEL_11_FEATURES:
        val = features14.get(name)
        if val is None:
            missing.append(name)
            vec.append(0.0)
        else:
            vec.append(float(val))
    return vec, missing


def infer_image_modality(image_array: np.ndarray) -> str:
    """
    Heuristic modality check for uploaded grayscale images.
    Returns: "mri", "xray", or "unknown".
    """
    arr = np.asarray(image_array, dtype=np.float32)
    if arr.ndim != 2:
        return "unknown"
    mean_v = float(arr.mean())
    std_v = float(arr.std())
    p95 = float(np.percentile(arr, 95))
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    edge = float(np.mean(np.abs(gx))) + float(np.mean(np.abs(gy)))

    # Typical X-ray: brighter highlights + stronger edge contrast.
    if mean_v > 0.45 and p95 > 0.85 and (std_v > 0.18 or edge > 0.22):
        return "xray"
    # Typical MRI in this app preprocessing: darker background and softer highlights.
    if mean_v < 0.38 and p95 < 0.8:
        return "mri"
    return "unknown"


def show_result(result: Dict[str, Any]):
    pred = result.get("prediction", result)
    scores = parse_scores(pred.get("predictions", []))
    st.subheader("Prediction")
    st.json(pred)
    mode = pred.get("inference_mode")
    if mode == "fallback":
        st.warning("Backend used fallback scoring, not the trained model. Check checkpoint/model paths.")
    elif mode == "model":
        st.success("Inference mode: trained model")

    if isinstance(scores, list) and scores:
        chart_data = {
            "Normal": float(scores[0]),
            "Abnormal": float(scores[1]) if len(scores) > 1 else float(scores[0]),
        }
        st.bar_chart(chart_data)

    guidance = result.get("guidance")
    if guidance:
        st.subheader("Exercise Guidance")
        st.json(guidance)


def call_layer2_advice(
    model_id: str,
    body_part: str,
    prediction_obj: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    scores = parse_scores(prediction_obj.get("predictions", []))
    if not isinstance(scores, list) or not scores:
        return None
    normal = float(scores[0])
    abnormal = float(scores[1]) if len(scores) > 1 else float(scores[0])
    prediction = prediction_obj.get("predicted_label", "abnormal" if abnormal >= normal else "normal")
    abnormal_probability = prediction_obj.get("abnormal_probability", abnormal)
    confidence = float(abnormal_probability if prediction == "abnormal" else max(0.0, min(1.0, 1.0 - float(abnormal_probability))))
    payload = {
        "model_id": model_id,
        "body_part": body_part,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            "normal": normal,
            "abnormal": abnormal,
        },
        "inference_mode": prediction_obj.get("inference_mode", "fallback"),
    }
    try:
        return api_post("/api/exercise-advice", payload)
    except Exception as e:
        st.error(f"Layer 2 advice request failed: {e}")
        return None


def render_domain_result_panels(domain: str, model_id: str, pred_result: Dict[str, Any]) -> None:
    pred = pred_result
    probs = parse_scores(pred.get("predictions", []))
    abnormal_prob = float(pred.get("abnormal_probability", probs[1] if isinstance(probs, list) and len(probs) > 1 else 0.0))
    label = str(pred.get("predicted_label", "abnormal" if abnormal_prob >= 0.5 else "normal"))
    confidence = abnormal_prob if label == "abnormal" else max(0.0, 1.0 - abnormal_prob)

    st.subheader("Primary Diagnosis")
    st.json(
        {
            "domain": domain,
            "model_id": model_id,
            "predicted_label": label,
            "confidence": round(float(confidence), 4),
            "inference_mode": pred.get("inference_mode", "fallback"),
        }
    )

    st.subheader("All Probabilities")
    st.json(
        {
            "normal": float(probs[0]) if isinstance(probs, list) and len(probs) > 0 else 0.0,
            "abnormal": float(probs[1]) if isinstance(probs, list) and len(probs) > 1 else abnormal_prob,
        }
    )

    if pred.get("abnormality_details") is not None:
        st.subheader("Abnormality Details")
        st.json(pred.get("abnormality_details"))

    advice = call_layer2_advice(model_id, domain, pred_result)
    if advice:
        st.subheader("Exercise Advice (Layer 2)")
        st.json(advice)


def validate_modality_or_stop(domain: str, arr: np.ndarray) -> None:
    payload = {
        "expected_domain": domain,
        "data": arr.flatten().tolist(),
        "shape": [1, 1, 64, 64],
    }
    try:
        res = api_post("/validate_modality", payload)
    except Exception as e:
        st.error(f"Modality validation failed: {e}")
        st.stop()

    if not bool(res.get("compatible", True)):
        st.error(f"Upload blocked: {res.get('reason', 'incompatible modality')}")
        st.stop()


try:
    models_by_domain = api_get("/models")
except Exception as e:
    st.error(f"API connection failed: {e}")
    st.stop()

try:
    perf = api_get("/model_performance")
    st.sidebar.markdown("### Selected Model Metrics")
    st.sidebar.json(perf)
except Exception:
    pass

all_models: Dict[str, Dict[str, str]] = {}
for domain, models in models_by_domain.items():
    for model_name in models.keys():
        all_models[model_name] = {"domain": domain}

if not all_models:
    st.warning("No models returned by API /models")
    st.stop()

tab_brain, tab_bone, tab_cell, tab_batch = st.tabs(["Brain", "Bone", "Cellular", "Batch"])

with tab_brain:
    st.subheader("Brain Inference")
    brain_models = [m for m, meta in all_models.items() if meta["domain"] == "brain"]
    if not brain_models:
        st.info("No brain models available")
    else:
        model = st.selectbox("Model", brain_models, key="brain_model")
        condition = st.text_input("Condition hint", "mri_abnormality", key="brain_condition")
        file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg", "bmp"], key="brain_upload")
        if st.button("Run Brain", key="run_brain"):
            if not file:
                st.warning("Upload an image first")
            else:
                img = Image.open(file).convert("L").resize((64, 64))
                arr = np.array(img, dtype=np.float32) / 255.0
                validate_modality_or_stop("brain", arr)
                payload = {
                    "model": model,
                    "data": arr.flatten().tolist(),
                    "shape": [1, 1, 64, 64],
                    "body_part_hint": "brain",
                    "condition_hint": condition,
                    "current_workout_intensity": INTENSITY,
                    "latest_biomarker_scores": {},
                }
                try:
                    pred_result = api_post(
                        "/predict",
                        {
                            "model": model,
                            "data": payload["data"],
                            "shape": payload["shape"],
                            "body_part_hint": payload["body_part_hint"],
                            "condition_hint": payload["condition_hint"],
                        },
                    )
                    render_domain_result_panels("brain", model, pred_result)
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab_bone:
    st.subheader("Bone Inference")
    bone_models = [m for m, meta in all_models.items() if meta["domain"] == "bone"]
    if not bone_models:
        st.info("No bone models available")
    else:
        model = st.selectbox("Model", bone_models, key="bone_model")
        part = st.text_input("Body part", "finger", key="bone_part")
        condition = st.text_input("Condition hint", "xray_abnormality", key="bone_condition")
        file = st.file_uploader("Upload X-ray", type=["png", "jpg", "jpeg", "bmp"], key="bone_upload")
        if st.button("Run Bone", key="run_bone"):
            if not file:
                st.warning("Upload an image first")
            else:
                img = Image.open(file).convert("L").resize((64, 64))
                arr = np.array(img, dtype=np.float32) / 255.0
                validate_modality_or_stop("bone", arr)
                payload = {
                    "model": model,
                    "data": arr.flatten().tolist(),
                    "shape": [1, 1, 64, 64],
                    "body_part_hint": part,
                    "condition_hint": condition,
                    "current_workout_intensity": INTENSITY,
                    "latest_biomarker_scores": {},
                }
                try:
                    pred_result = api_post(
                        "/predict",
                        {
                            "model": model,
                            "data": payload["data"],
                            "shape": payload["shape"],
                            "body_part_hint": payload["body_part_hint"],
                            "condition_hint": payload["condition_hint"],
                        },
                    )
                    render_domain_result_panels("bone", model, pred_result)
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab_cell:
    st.subheader("Cellular Inference")
    if parse_report is None or predict_cellular is None or map_accepted_records_to_features is None:
        st.error("Cellular advanced pipeline modules not available. Check imports/install.")
    else:
        report_files = st.file_uploader(
            "Upload Report (single or multiple files)",
            type=["pdf", "jpg", "jpeg", "png", "tiff", "bmp"],
            accept_multiple_files=True,
            key="cell_report_upload_multi",
        )
        if st.button("Run Cellular", key="run_cell_new"):
            if not report_files:
                st.warning("Upload one or more report files first.")
            else:
                temp_paths = []
                try:
                    for rf in report_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(rf.name).suffix) as tf:
                            tf.write(rf.getbuffer())
                            temp_paths.append(tf.name)

                    parsed = parse_report(temp_paths if len(temp_paths) > 1 else temp_paths[0])
                    st.write("Merged OCR JSON")
                    st.json(parsed)
                    st.success(f"Total: {len(parsed.get('records', []))} records extracted from {len(report_files)} file(s)")

                    accepted = parsed.get("accepted_records", [])
                    features = map_accepted_records_to_features(accepted)
                    st.subheader("Cellular Model Input Features (11)")
                    st.json(features)
                    pred = predict_cellular(features)

                    primary = pred["primary_condition"]
                    disease_id = primary["disease_id"]
                    disease_meta = DISEASE_REGISTRY.get(disease_id, {})

                    st.subheader("Primary Diagnosis")
                    st.json(primary)
                    if disease_meta:
                        st.write(f"ICD-10: {disease_meta.get('icd10_code')}")
                        st.write(disease_meta.get("description", ""))
                        urgency = str(disease_meta.get("urgency", "routine"))
                        if due_date_from_urgency is not None:
                            st.write(f"See doctor by: {due_date_from_urgency(urgency)}")

                    st.subheader("All Conditions")
                    st.json({
                        "secondary_conditions": pred.get("secondary_conditions", []),
                        "all_probabilities": pred.get("all_probabilities", {}),
                        "overall_status": pred.get("overall_status"),
                        "risk_level": pred.get("risk_level"),
                    })

                    st.subheader("Biomarker Flags")
                    st.json(pred.get("biomarker_flags", []))

                    if disease_meta:
                        st.subheader("Lifestyle Tips")
                        for tip in disease_meta.get("lifestyle_tips", []):
                            st.write(f"- {tip}")

                    advice_payload = {
                        "model_id": "cellular_risk",
                        "body_part": "cellular",
                        "prediction": "abnormal" if pred.get("overall_status") == "abnormal" else "normal",
                        "confidence": float(primary.get("confidence", 0.0)),
                        "probabilities": {
                            "normal": float(pred.get("all_probabilities", {}).get("healthy", 0.0)),
                            "abnormal": float(1.0 - float(pred.get("all_probabilities", {}).get("healthy", 0.0))),
                        },
                        "inference_mode": pred.get("inference_mode", "model"),
                        "disease_id": disease_id,
                    }

                    st.subheader("Exercise Advice (Layer 2)")
                    try:
                        advice = api_post("/api/exercise-advice", advice_payload)
                        st.json(advice)
                    except Exception as e:
                        st.warning(f"Exercise advice API unavailable: {e}")
                        st.json(advice_payload)

                except Exception as e:
                    st.error(f"Cellular pipeline failed: {e}")
                finally:
                    for p in temp_paths:
                        try:
                            os.remove(p)
                        except Exception:
                            pass

with tab_batch:
    st.subheader("Batch Prediction")
    model = st.selectbox("Model", list(all_models.keys()), key="batch_model")
    payload_text = st.text_area(
        "Batch payload JSON",
        value=json.dumps({"model": model, "samples": [[0.1, 0.2], [0.3, 0.4]], "shape": [2]}, indent=2),
        height=220,
    )
    if st.button("Run Batch", key="run_batch"):
        try:
            payload = json.loads(payload_text)
            payload["model"] = model
            result = api_post("/predict_batch", payload)
            st.json(result)
        except Exception as e:
            st.error(f"Batch request failed: {e}")
