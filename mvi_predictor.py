"""Compatibility adapter used by the lightweight Streamlit web app."""

from auc869_web_predictor import AUC869WebPredictor


class MVIPredictor(AUC869WebPredictor):
    """Preserve the original web-app import while using the AUC869 model."""

