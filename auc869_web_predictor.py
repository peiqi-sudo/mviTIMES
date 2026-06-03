"""Lightweight web predictor derived from the exploratory AUC869 teaching model.

The training routine intentionally reproduces the teaching model's label-aware
Parent selection. Inference never reads MVI.State: every available Parent is
scored independently and the highest Parent probability is reported for each
Image. This is an exploratory web prototype, not a clinically validated model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier


SEED = 20260529
PARENT_AREA_MM2 = 1.0
DISTANCE_RADIUS_UM = 50
FALLBACK_NND_UM = 1000 * np.sqrt(2)
VALID_PARENTS = ["tc1", "tc2", "tc3"]

TARGET_CELLTYPES = {
    "CD34.Endothelial": "CD34Endo",
    "CD56.NK": "CD56NK",
    "CXCL12.CD163.M2": "M2",
    "CXCR4.CD56.NK": "CXCR4NK",
    "GPC3.Tumor": "GPC3Tumor",
}

SELECTED_FEATURES = [
    "prop_CXCR4NK",
    "M2_to_CXCR4NK_density_within_50",
    "cxcr4_nk_ratio_total_nk",
    "GPC3Tumor_to_CXCR4NK_density_within_50",
    "prop_CD56NK",
    "m2_to_total_myeloid_like_ratio",
    "CD34Endo_to_CXCR4NK_density_within_50",
    "CXCR4NK_to_CD56NK_frac_within_50",
    "CXCR4NK_to_CD56NK_density_within_50",
    "density_CXCR4NK",
    "density_CD56NK",
    "CXCR4NK_to_GPC3Tumor_frac_within_50",
    "CD56NK_to_CXCR4NK_density_within_50",
    "CD34Endo_to_CXCR4NK_median_nnd",
    "CXCR4NK_to_CD34Endo_mean_nnd",
    "CXCR4NK_to_CD34Endo_median_nnd",
    "CXCR4NK_to_GPC3Tumor_density_within_50",
    "CXCR4NK_to_CD34Endo_frac_within_50",
    "CD34Endo_to_CXCR4NK_mean_nnd",
    "GPC3Tumor_to_CD56NK_density_within_50",
]

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": 1,
}

MODEL_WARNING = (
    "Exploratory web prototype derived from the AUC869 teaching model. "
    "Training Parent selection reads true MVI.State. Web inference does not "
    "read MVI.State: it scores each uploaded Parent and reports "
    "the maximum Parent probability. The retrospective teaching AUC=0.869 "
    "must not be interpreted as validated web-deployment performance."
)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


def _clean_input(raw_df: pd.DataFrame, require_labels: bool) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [
        str(column).replace("\ufeff", "").strip().strip('"').strip("'")
        for column in df.columns
    ]
    required = {"Image", "Parent", "x.axis", "y.axis", "CellType"}
    if require_labels:
        required.add("MVI.State")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in ["Image", "Parent", "CellType"]:
        df[column] = df[column].astype("string").str.strip()
    if "MVI.State" in df.columns:
        df["MVI.State"] = df["MVI.State"].astype("string").str.strip()

    if require_labels:
        # The teaching cohort is explicitly defined by tc1/tc2/tc3.
        df = df[df["Parent"].isin(VALID_PARENTS)].copy()
    else:
        # Web uploads may use names such as tumor1 or Core_Tumor. Each uploaded
        # Parent must still represent an approximately 1 mm2 analysis region.
        df = df[df["Parent"].notna() & df["Parent"].ne("")].copy()
    df["x_um"] = pd.to_numeric(df["x.axis"], errors="coerce")
    df["y_um"] = pd.to_numeric(df["y.axis"], errors="coerce")
    drop_columns = ["Image", "Parent", "CellType", "x_um", "y_um"]
    if require_labels:
        drop_columns.append("MVI.State")
    df = df.dropna(subset=drop_columns)
    if df.empty:
        raise ValueError("No valid cells remain after input cleaning.")
    return df


def _compute_directional_pair_features(
    coords_by_short: dict[str, np.ndarray],
    source_short: str,
    target_short: str,
) -> dict[str, float]:
    source_coords = coords_by_short[source_short]
    target_coords = coords_by_short[target_short]
    prefix = f"{source_short}_to_{target_short}"
    result = {
        f"{prefix}_mean_nnd": FALLBACK_NND_UM,
        f"{prefix}_median_nnd": FALLBACK_NND_UM,
        f"{prefix}_frac_within_50": 0.0,
        f"{prefix}_density_within_50": 0.0,
    }
    if len(source_coords) == 0 or len(target_coords) == 0:
        return result

    tree = cKDTree(target_coords)
    nearest_distances, _ = tree.query(source_coords, k=1)
    neighbor_lists = tree.query_ball_point(source_coords, r=DISTANCE_RADIUS_UM)
    neighbor_counts = np.asarray([len(values) for values in neighbor_lists], dtype=float)
    radius_area_mm2 = np.pi * DISTANCE_RADIUS_UM**2 / 1_000_000

    result[f"{prefix}_mean_nnd"] = float(np.mean(nearest_distances))
    result[f"{prefix}_median_nnd"] = float(np.median(nearest_distances))
    result[f"{prefix}_frac_within_50"] = float(np.mean(neighbor_counts > 0))
    result[f"{prefix}_density_within_50"] = _safe_divide(
        np.mean(neighbor_counts),
        radius_area_mm2,
    )
    return result


def _compute_one_parent_features(
    parent_df: pd.DataFrame,
    image: str,
    parent: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "Image": image,
        "Parent": parent,
        "area_mm2": PARENT_AREA_MM2,
    }
    if "MVI.State" in parent_df.columns and parent_df["MVI.State"].notna().any():
        row["MVI.State"] = parent_df["MVI.State"].dropna().iloc[0]

    counts = parent_df["CellType"].value_counts().to_dict()
    total_cells = len(parent_df)
    coords_by_short: dict[str, np.ndarray] = {}
    for raw_name, short_name in TARGET_CELLTYPES.items():
        cell_df = parent_df[parent_df["CellType"] == raw_name]
        n_cells = int(counts.get(raw_name, 0))
        row[f"density_{short_name}"] = _safe_divide(n_cells, PARENT_AREA_MM2)
        row[f"prop_{short_name}"] = _safe_divide(n_cells, total_cells)
        coords_by_short[short_name] = cell_df[["x_um", "y_um"]].to_numpy(dtype=float)

    n_cd56_nk = int(counts.get("CD56.NK", 0))
    n_cxcr4_nk = int(counts.get("CXCR4.CD56.NK", 0))
    n_m2 = int(counts.get("CXCL12.CD163.M2", 0))
    total_nk = n_cd56_nk + n_cxcr4_nk
    row["total_nk_density"] = _safe_divide(total_nk, PARENT_AREA_MM2)
    row["cxcr4_nk_ratio_total_nk"] = _safe_divide(n_cxcr4_nk, total_nk)
    # Kept for exact teaching-model compatibility. This is M2 / (M2 + CXCR4NK).
    row["m2_to_total_myeloid_like_ratio"] = _safe_divide(n_m2, n_m2 + n_cxcr4_nk)

    short_names = list(TARGET_CELLTYPES.values())
    for source_short in short_names:
        for target_short in short_names:
            if source_short != target_short:
                row.update(
                    _compute_directional_pair_features(
                        coords_by_short,
                        source_short,
                        target_short,
                    )
                )
    return row


def extract_parent_features(
    raw_df: pd.DataFrame,
    require_labels: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convert cell coordinates into one feature row per Image + Parent."""
    df = _clean_input(raw_df, require_labels=require_labels)
    groups = df.groupby(["Image", "Parent"], sort=True)
    rows = []
    for index, ((image, parent), parent_df) in enumerate(groups, start=1):
        rows.append(_compute_one_parent_features(parent_df, image, parent))
        if verbose and (index % 50 == 0 or index == groups.ngroups):
            print(f"Parent feature extraction: {index} / {groups.ngroups}")
    return pd.DataFrame(rows).sort_values(["Image", "Parent"]).reset_index(drop=True)


def _rank01(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=ascending)


def _add_interaction_scores(parent_df: pd.DataFrame) -> pd.DataFrame:
    df = parent_df.copy()
    eps = 1e-9
    df["M2_to_CXCR4NK_density_within_50_log1p"] = np.log1p(
        df["M2_to_CXCR4NK_density_within_50"].clip(lower=0)
    )
    df["density_CXCR4NK_log1p"] = np.log1p(df["density_CXCR4NK"].clip(lower=0))
    df["inv_CXCR4NK_to_M2_nnd"] = 1.0 / np.log1p(
        df["CXCR4NK_to_M2_mean_nnd"].clip(lower=0) + eps
    )
    df["NK_M2_interaction_score_50"] = (
        df["density_CXCR4NK_log1p"]
        * df["M2_to_CXCR4NK_density_within_50_log1p"]
        * df["inv_CXCR4NK_to_M2_nnd"]
    )
    return df


def _select_label_aware_training_parents(parent_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the retrospective teaching selection for model export only."""
    if "MVI.State" not in parent_df.columns:
        raise ValueError("Training Parent selection requires MVI.State.")
    df = _add_interaction_scores(parent_df)
    selected_rows = []
    for _, group in df.groupby("Image", sort=True):
        group = group.copy()
        state = group["MVI.State"].iloc[0]
        hotspot = (
            _rank01(group["cxcr4_nk_ratio_total_nk"])
            + _rank01(group["density_CXCR4NK"])
            + _rank01(group["M2_to_CXCR4NK_density_within_50"])
            + _rank01(group["NK_M2_interaction_score_50"])
            + _rank01(-group["CXCR4NK_to_M2_mean_nnd"])
        )
        if state == "MVI.pos":
            selected_index = hotspot.idxmax()
        else:
            nonzero = group[group["M2_to_CXCR4NK_density_within_50"] > 0].copy()
            if len(nonzero):
                cold = (
                    _rank01(nonzero["M2_to_CXCR4NK_density_within_50"])
                    + _rank01(nonzero["NK_M2_interaction_score_50"])
                    + 0.25 * _rank01(nonzero["density_CXCR4NK"])
                )
                selected_index = cold.idxmin()
            else:
                selected_index = hotspot.idxmin()
        selected_rows.append(group.loc[selected_index])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


class AUC869WebPredictor:
    """Small loader, trainer and inference adapter for CSV-based web scoring."""

    def __init__(self) -> None:
        self.model: ExtraTreesClassifier | None = None
        self.feature_columns = SELECTED_FEATURES.copy()
        self.metadata: dict[str, object] = {}

    def train_and_save(
        self,
        train_csv_path: str | Path,
        model_save_path: str | Path,
    ) -> dict[str, object]:
        """Train the full-cohort teaching-derived prototype and save a pkl."""
        raw_df = pd.read_csv(train_csv_path, low_memory=False)
        parent_df = extract_parent_features(raw_df, require_labels=True, verbose=True)
        train_df = _select_label_aware_training_parents(parent_df)
        y_train = train_df["MVI.State"].eq("MVI.pos").astype(int)

        self.model = ExtraTreesClassifier(**MODEL_PARAMS)
        self.model.fit(train_df[self.feature_columns], y_train)
        self.metadata = {
            "format_version": "mviTIMES_AUC869_web_lite_v1",
            "model_family": "ExtraTreesClassifier",
            "model_params": MODEL_PARAMS,
            "features": self.feature_columns,
            "n_features": len(self.feature_columns),
            "n_training_images": int(train_df["Image"].nunique()),
            "training_parent_selection": "label_aware_teaching_rule",
            "web_inference_parent_strategy": "score_each_uploaded_parent_then_take_max_probability",
            "web_parent_area_requirement": "each uploaded Parent should represent approximately 1 mm2",
            "threshold": 0.5,
            "score_semantics": "exploratory_risk_score_not_calibrated_clinical_probability",
            "warning": MODEL_WARNING,
        }
        payload = {
            "model": self.model,
            "features": self.feature_columns,
            "metadata": self.metadata,
        }
        model_save_path = Path(model_save_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, model_save_path)
        return self.metadata

    def load_model(self, model_path: str | Path) -> None:
        payload = joblib.load(model_path)
        missing = sorted({"model", "features", "metadata"} - set(payload))
        if missing:
            raise ValueError(f"Model package is missing keys: {missing}")
        self.model = payload["model"]
        self.feature_columns = list(payload["features"])
        self.metadata = dict(payload["metadata"])

    def predict_parent_scores(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Load or train a model before prediction.")
        parent_df = extract_parent_features(raw_df, require_labels=False, verbose=False)
        parent_df["Parent_mviTIMES_Score"] = self.model.predict_proba(
            parent_df[self.feature_columns]
        )[:, 1]
        return parent_df

    def predict_score(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Return one label-blind hotspot-max score per Image."""
        parent_df = self.predict_parent_scores(raw_df)
        rows = []
        for image, group in parent_df.groupby("Image", sort=True):
            ordered = group.sort_values(
                ["Parent_mviTIMES_Score", "Parent"],
                ascending=[False, True],
            )
            representative = ordered.iloc[0]
            score = float(representative["Parent_mviTIMES_Score"])
            rows.append(
                {
                    "Image_ID": image,
                    "mviTIMES_Score": score,
                    "Predicted_MVI_State": "MVI.pos" if score >= 0.5 else "MVI.neg",
                    "Representative_Parent": representative["Parent"],
                    "Parent_Count": int(len(group)),
                    "Parent_Score_Min": float(group["Parent_mviTIMES_Score"].min()),
                    "Parent_Score_Mean": float(group["Parent_mviTIMES_Score"].mean()),
                    "Parent_Score_Max": float(group["Parent_mviTIMES_Score"].max()),
                }
            )
        return pd.DataFrame(rows).sort_values(
            "mviTIMES_Score",
            ascending=False,
        ).reset_index(drop=True)


def model_feature_meanings() -> Iterable[tuple[str, str]]:
    """Yield compact descriptions suitable for a web documentation page."""
    meanings = {
        "prop_CXCR4NK": "CXCR4+ CD56+ NK proportion among all cells",
        "M2_to_CXCR4NK_density_within_50": "local CXCR4+ CD56+ NK density within 50 um around M2 cells",
        "cxcr4_nk_ratio_total_nk": "CXCR4+ CD56+ NK fraction among total NK cells",
        "m2_to_total_myeloid_like_ratio": "legacy name: M2 / (M2 + CXCR4+ CD56+ NK)",
    }
    for feature in SELECTED_FEATURES:
        yield feature, meanings.get(feature, "Teaching-model spatial feature")
