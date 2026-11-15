"""
SINYAL · EarlySkill-ID | Context-Aware Early Detection of Emerging IT Skills
Server Aplikasi Web Prediksi Berbasis Hasil Model Notebook (4 Fitur Inti · Dual-Corpus: Indonesia & Global)
"""
import json
import os
import pickle
import sqlite3
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import numpy as np


# ==============================================================================
# Definisi Kelas Estimator Native agar Kompatibel saat Unpickling Model (.pkl)
# ==============================================================================
def _expit(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=np.float64), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


class CausalPreprocessor:
    def __init__(self, scale: bool = True):
        self.scale = scale
        self.medians_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.medians_ is not None:
            X_imp = np.where(np.isnan(X_arr), self.medians_[None, :], X_arr)
        else:
            X_imp = X_arr.copy()
        X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scale and self.means_ is not None and self.stds_ is not None:
            return np.clip((X_imp - self.means_[None, :]) / self.stds_[None, :], -8.0, 8.0)
        return X_imp


class NativeBalancedLogisticRegression:
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state
        self.prep = CausalPreprocessor(scale=True)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s = self.prep.transform(X)
        p1 = _expit(X_s @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p1, p1])


class _FastHistogramTree:
    def __init__(self, **kwargs: Any):
        self.nodes: List[Dict[str, Any]] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)

        def _traverse(node_id: int, mask: np.ndarray) -> None:
            if not np.any(mask):
                return
            nd = self.nodes[node_id]
            if nd["is_leaf"]:
                out[mask] = nd["value"]
                return
            f = nd["feature"]
            th = nd["raw_thresh"]
            go_left = mask & (X[:, f] <= th)
            go_right = mask & (X[:, f] > th)
            _traverse(nd["left"], go_left)
            _traverse(nd["right"], go_right)

        _traverse(0, np.ones(n, dtype=bool))
        return out


class NativeRandomForestClassifier:
    def __init__(self, **kwargs: Any):
        self.prep = CausalPreprocessor(scale=False)
        self.trees: List[_FastHistogramTree] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.prep.transform(X)
        preds = np.mean([t.predict(X_imp) for t in self.trees], axis=0)
        p1 = np.clip(preds, 1e-5, 1.0 - 1e-5)
        return np.column_stack([1.0 - p1, p1])


class NativeGradientBoostingClassifier:
    def __init__(self, **kwargs: Any):
        self.learning_rate: float = 0.05
        self.prep = CausalPreprocessor(scale=False)
        self.base_logit_: float = 0.0
        self.trees: List[_FastHistogramTree] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.prep.transform(X)
        logits = np.full(X_imp.shape[0], self.base_logit_, dtype=np.float64)
        for t in self.trees:
            logits += self.learning_rate * t.predict(X_imp)
        p1 = _expit(logits)
        return np.column_stack([1.0 - p1, p1])


_CUSTOM_CLASSES = {
    "CausalPreprocessor": CausalPreprocessor,
    "NativeBalancedLogisticRegression": NativeBalancedLogisticRegression,
    "_FastHistogramTree": _FastHistogramTree,
    "NativeRandomForestClassifier": NativeRandomForestClassifier,
    "NativeGradientBoostingClassifier": NativeGradientBoostingClassifier,
}

for _cls_name, _cls_obj in _CUSTOM_CLASSES.items():
    setattr(sys.modules["__main__"], _cls_name, _cls_obj)


class _SafeModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name in _CUSTOM_CLASSES:
            return _CUSTOM_CLASSES[name]
        return super().find_class(module, name)


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return _SafeModelUnpickler(f).load()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CORPUS_CONFIG = {
    "INDONESIA": {
        "db_path": os.path.join(BASE_DIR, "earlyskill_id_indo.db"),
        "pkl_path": os.path.join(BASE_DIR, "trained_models_indo_4fitur.pkl"),
        "bundle_path": os.path.join(BASE_DIR, "webapp_bundle_indo_4fitur.json"),
        "default_quarter": "2026-Q4",
    },
    "GLOBAL": {
        "db_path": os.path.join(BASE_DIR, "earlyskill_id.db"),
        "pkl_path": os.path.join(BASE_DIR, "trained_models_4fitur.pkl"),
        "bundle_path": os.path.join(BASE_DIR, "webapp_bundle_4fitur.json"),
        "default_quarter": "2026-Q1",
    },
}

MODEL_ARTIFACTS_BY_SCOPE: Dict[str, Dict[str, Any]] = {}
WEBAPP_BUNDLES_BY_SCOPE: Dict[str, Dict[str, Any]] = {}

for _scope, _cfg in CORPUS_CONFIG.items():
    if os.path.exists(_cfg["pkl_path"]):
        MODEL_ARTIFACTS_BY_SCOPE[_scope] = _load_pickle(_cfg["pkl_path"])
    if os.path.exists(_cfg["bundle_path"]):
        with open(_cfg["bundle_path"], "r", encoding="utf-8") as _fb:
            WEBAPP_BUNDLES_BY_SCOPE[_scope] = json.load(_fb)


CORE_4_FEATURES = [
    "qoq_growth_smoothed",
    "ema_ratio_short_long",
    "cooc_weighted_degree_roll4",
    "role_entropy_roll4",
]


def _resolve_scope(qs: Dict[str, List[str]]) -> str:
    raw = (qs.get("scope", ["INDONESIA"])[0] or "INDONESIA").upper()
    return "GLOBAL" if raw in ("GLOBAL", "GLOB") else "INDONESIA"


class EarlySkillRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join(BASE_DIR, "static", "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        if parsed.path == "/api/meta":
            scope = _resolve_scope(qs)
            bundle = WEBAPP_BUNDLES_BY_SCOPE.get(scope, {})
            artifacts = MODEL_ARTIFACTS_BY_SCOPE.get(scope, {})
            sel_info = artifacts.get("selected_model_info", {})
            safe_sel_info = {
                k: v
                for k, v in sel_info.items()
                if k not in ("fitted_estimator", "ranking_table")
            }
            self._send_json({
                "scope": scope,
                "meta": bundle.get("meta", {}),
                "core_4_features": CORE_4_FEATURES,
                "feature_sets": artifacts.get("feature_sets", {}),
                "risk_set_thresholds": artifacts.get("risk_set_thresholds", {}),
                "selected_model_info": safe_sel_info,
            })
            return

        if parsed.path == "/api/watchlist":
            scope = _resolve_scope(qs)
            cfg_scope = CORPUS_CONFIG[scope]
            quarter = qs.get("quarter", [cfg_scope["default_quarter"]])[0]
            model_code = qs.get("model_code", ["M4_Demand_Context_Role"])[0]
            algorithm = qs.get("algorithm", ["RandomForest"])[0]

            conn = sqlite3.connect(cfg_scope["db_path"])
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    q.skill,
                    q.domain_category,
                    q.year_quarter,
                    q.quarter_idx,
                    q.job_count,
                    q.total_jobs_quarter,
                    q.skill_share,
                    q.qoq_growth_smoothed,
                    q.ema_ratio_short_long,
                    q.cooc_weighted_degree_roll4,
                    q.role_entropy_roll4,
                    q.rolling_mean_4q,
                    q.causal_cummax_share,
                    q.in_risk_set,
                    q.target_emerging,
                    q.label_reason,
                    p.model_code,
                    p.algorithm,
                    p.fold_id,
                    p.fold_role,
                    p.pred_prob AS predicted_prob,
                    p.threshold_used AS oof_threshold,
                    p.pred_label AS early_warning_flag,
                    p.quarterly_rank
                FROM Skill_Quarter_Panel q
                LEFT JOIN Predictions_SHAP_LeadTime p
                    ON q.skill = p.skill
                    AND q.year_quarter = p.year_quarter
                    AND p.model_code = ?
                    AND p.algorithm = ?
                WHERE q.year_quarter = ? AND q.in_risk_set = 1
                ORDER BY COALESCE(p.pred_prob, 0.0) DESC, q.job_count DESC
                """,
                (model_code, algorithm, quarter),
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            self._send_json({
                "scope": scope,
                "quarter": quarter,
                "model_code": model_code,
                "algorithm": algorithm,
                "count": len(rows),
                "items": rows,
            })
            return

        if parsed.path == "/api/predict":
            scope = _resolve_scope(qs)
            model_code = qs.get("model_code", ["M4_Demand_Context_Role"])[0]
            algorithm = qs.get("algorithm", ["RandomForest"])[0]

            feat_map = {
                "qoq_growth_smoothed": float(qs.get("qoq_growth_smoothed", [0.25])[0]),
                "ema_ratio_short_long": float(qs.get("ema_ratio_short_long", [1.15])[0]),
                "cooc_weighted_degree_roll4": float(qs.get("cooc_weighted_degree_roll4", [2.50])[0]),
                "role_entropy_roll4": float(qs.get("role_entropy_roll4", [0.45])[0]),
            }

            artifacts = MODEL_ARTIFACTS_BY_SCOPE.get(scope) or MODEL_ARTIFACTS_BY_SCOPE["GLOBAL"]
            models_dict = artifacts.get("live_trained_models") or artifacts.get("trained_models_last_fold", {})
            key = (model_code, algorithm)
            if key not in models_dict:
                key = ("M4_Demand_Context_Role", "RandomForest")

            estimator, feat_names, tau_oof = models_dict[key]
            X = np.array([[feat_map[f] for f in feat_names]], dtype=float)
            prob = float(estimator.predict_proba(X)[0, 1])
            tau = float(tau_oof)

            self._send_json({
                "scope": scope,
                "model_code": key[0],
                "algorithm": key[1],
                "features": feat_map,
                "predicted_prob": round(prob, 4),
                "tau_oof": round(tau, 4),
                "early_warning_flag": int(prob >= tau),
            })
            return

        if parsed.path == "/api/provenance":
            scope = _resolve_scope(qs)
            cfg_scope = CORPUS_CONFIG[scope]
            skill = qs.get("skill", ["RAG"])[0]
            limit = min(100, max(1, int(qs.get("limit", [25])[0])))

            conn = sqlite3.connect(cfg_scope["db_path"])
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    p.job_id,
                    p.year_quarter,
                    p.canonical_role,
                    p.canonical_skill,
                    p.domain_category,
                    p.extraction_source,
                    r.created_dt,
                    r.company_clean,
                    r.location_clean,
                    r.title_clean,
                    r.canonical_skills
                FROM Job_Skill_Provenance p
                JOIN Raw_Job_Postings r ON p.job_id = r.job_id
                WHERE p.canonical_skill = ?
                ORDER BY p.year_quarter DESC, p.job_id DESC
                LIMIT ?
                """,
                (skill, limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            self._send_json({
                "scope": scope,
                "skill": skill,
                "count": len(rows),
                "items": rows,
            })
            return

        return super().do_GET()

    def _send_json(self, obj: Any) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"SINYAL · EarlySkill-ID berjalan di http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), EarlySkillRequestHandler).serve_forever()
