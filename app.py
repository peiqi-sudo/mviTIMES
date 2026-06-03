"""International Streamlit interface for the mviTIMES AUC869 platform."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from scipy import ndimage
from skimage import feature, filters, measure, morphology, segmentation

from mvi_predictor import MVIPredictor


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mvi_scoring_model.pkl"
LOGO_PATH = BASE_DIR / "logo.png"
TUTORIAL_PATH = BASE_DIR / "tutorial.mp4"

REQUIRED_CSV_COLUMNS = ["Image", "Parent", "x.axis", "y.axis", "CellType"]
CORE_CELL_TYPES = [
    "CD34.Endothelial",
    "CD56.NK",
    "CXCL12.CD163.M2",
    "CXCR4.CD56.NK",
    "GPC3.Tumor",
]
MODEL_THRESHOLD = 0.5


st.set_page_config(
    page_title="mviTIMES | Spatial MVI Risk Explorer",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #0B233F;
        --blue: #145A8D;
        --teal: #0B8B8C;
        --ice: #EEF6F8;
        --slate: #526476;
        --line: #DCE6EC;
        --risk: #B84545;
        --low: #2573A6;
    }
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #F6FAFC 0%, #FFFFFF 28rem);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071B31 0%, #0B2948 100%);
        border-right: 1px solid rgba(255,255,255,.10);
    }
    [data-testid="stSidebar"] * { color: #E7F3F8 !important; }
    [data-testid="stSidebar"] img {
        background: white;
        border-radius: 50%;
        padding: .35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .45rem .25rem;
    }
    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 { color: var(--navy); letter-spacing: -.02em; }
    .eyebrow {
        color: var(--teal);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }
    .hero-title {
        color: var(--navy);
        font-size: clamp(2.55rem, 5vw, 4.6rem);
        font-weight: 800;
        letter-spacing: -.065em;
        line-height: .98;
        max-width: 930px;
        margin-bottom: 1.1rem;
    }
    .hero-subtitle {
        color: var(--slate);
        font-size: 1.17rem;
        line-height: 1.75;
        max-width: 850px;
        margin-bottom: 1.55rem;
    }
    .pill {
        display: inline-block;
        color: #0A5361;
        background: #E2F4F3;
        border: 1px solid #B8E1DF;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .04em;
        margin: 0 .35rem .45rem 0;
        padding: .34rem .68rem;
    }
    .section-heading {
        color: var(--navy);
        font-size: 1.65rem;
        font-weight: 760;
        letter-spacing: -.035em;
        margin: 1.7rem 0 .35rem;
    }
    .section-copy {
        color: var(--slate);
        line-height: 1.75;
        max-width: 900px;
        margin-bottom: 1rem;
    }
    .card {
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(16, 55, 78, .055);
        min-height: 178px;
        padding: 1.2rem 1.25rem;
        margin-bottom: .7rem;
    }
    .card-kicker {
        color: var(--teal);
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }
    .card-title {
        color: var(--navy);
        font-size: 1.08rem;
        font-weight: 760;
        margin-bottom: .45rem;
    }
    .card-copy { color: var(--slate); font-size: .91rem; line-height: 1.65; }
    .step {
        border-left: 3px solid var(--teal);
        color: var(--slate);
        min-height: 100px;
        padding: .2rem .2rem .25rem .85rem;
    }
    .step-number { color: var(--teal); font-size: .73rem; font-weight: 800; letter-spacing: .12em; }
    .step-title { color: var(--navy); font-size: 1rem; font-weight: 740; margin: .2rem 0; }
    .step-copy { font-size: .84rem; line-height: 1.45; }
    .notice {
        background: #F2F8FA;
        border: 1px solid #D7E7EB;
        border-left: 4px solid var(--teal);
        border-radius: 9px;
        color: #35566A;
        font-size: .9rem;
        line-height: 1.62;
        margin: .8rem 0;
        padding: .8rem 1rem;
    }
    .risk-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 15px;
        box-shadow: 0 8px 22px rgba(16, 55, 78, .055);
        padding: 1.2rem 1.25rem;
    }
    .risk-label { color: var(--slate); font-size: .75rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
    .risk-value { color: var(--navy); font-size: 2.6rem; font-weight: 820; letter-spacing: -.06em; margin: .25rem 0; }
    .risk-band { background: linear-gradient(90deg, #2573A6 0%, #2A9D8F 50%, #D25B55 100%); border-radius: 99px; height: 10px; margin: .65rem 0 .25rem; position: relative; }
    .risk-dot { background: #071B31; border: 2px solid white; border-radius: 50%; box-shadow: 0 1px 5px rgba(0,0,0,.25); height: 18px; position: absolute; top: -4px; transform: translateX(-50%); width: 18px; }
    .risk-scale { color: #6C7D88; display: flex; font-size: .72rem; justify-content: space-between; }
    .small-note { color: #657786; font-size: .78rem; line-height: 1.55; }
    .sidebar-brand {
        color: white;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -.03em;
        margin-top: .65rem;
    }
    .sidebar-copy { color: #B9D4E4; font-size: .78rem; line-height: 1.55; }
    .sidebar-tag {
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 8px;
        color: #D5E9F3;
        font-size: .73rem;
        margin-bottom: .45rem;
        padding: .45rem .52rem;
    }
    div.stButton > button {
        background: linear-gradient(100deg, #0E557F, #0B8B8C);
        border: 0;
        border-radius: 8px;
        color: white;
        font-weight: 700;
        min-height: 2.7rem;
    }
    div.stDownloadButton > button {
        border-color: #BFD5DF;
        border-radius: 8px;
        color: #174D6D;
        font-weight: 650;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: .7rem .85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_heading(title: str, copy: str = "") -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


def render_notice(text: str) -> None:
    st.markdown(f'<div class="notice">{text}</div>', unsafe_allow_html=True)


def render_feature_card(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-kicker">{kicker}</div>
            <div class="card-title">{title}</div>
            <div class="card-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step(number: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="step">
            <div class="step-number">STEP {number}</div>
            <div class="step-title">{title}</div>
            <div class="step-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_csv_safely(file) -> pd.DataFrame:
    for encoding in ["utf-8", "utf-8-sig", "gb18030", "gbk"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("The CSV encoding could not be parsed. Please upload a UTF-8 CSV file.")


def validate_coordinate_csv(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_CSV_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("The uploaded CSV contains no rows.")


@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str) -> MVIPredictor:
    predictor = MVIPredictor()
    predictor.load_model(model_path)
    return predictor


def risk_category(score: float) -> tuple[str, str]:
    if score >= MODEL_THRESHOLD:
        return "Higher exploratory risk", "#B84545"
    return "Lower exploratory risk", "#2573A6"


def render_risk_card(sample_id: str, score: float, parent: str) -> None:
    category, color = risk_category(score)
    percent = min(max(score, 0.0), 1.0) * 100
    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-label">{sample_id} &nbsp; | &nbsp; representative region: {parent}</div>
            <div class="risk-value">{score:.3f}</div>
            <div style="font-weight:750;color:{color};">{category}</div>
            <div class="risk-band"><div class="risk-dot" style="left:{percent:.2f}%;"></div></div>
            <div class="risk-scale"><span>0.0</span><span>exploratory cutoff: 0.5</span><span>1.0</span></div>
            <div class="small-note" style="margin-top:.75rem;">
                This score is a research-use risk index. It is not a calibrated clinical probability
                and should not be interpreted as a standalone diagnostic result.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(results_df: pd.DataFrame, download_name: str = "mviTIMES_scores.csv") -> None:
    if results_df.empty:
        st.warning("No score could be generated from the uploaded data.")
        return

    section_heading(
        "Exploratory Risk Report",
        "Patient-level mviTIMES scores are generated from the highest-scoring uploaded Parent region for each Image.",
    )
    higher_risk_n = int((results_df["mviTIMES_Score"] >= MODEL_THRESHOLD).sum())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Scored images", f"{len(results_df)}")
    metric_cols[1].metric("Higher-risk images", f"{higher_risk_n}")
    metric_cols[2].metric("Median score", f"{results_df['mviTIMES_Score'].median():.3f}")
    metric_cols[3].metric("Maximum score", f"{results_df['mviTIMES_Score'].max():.3f}")

    selected_image = st.selectbox(
        "Inspect an individual image",
        options=results_df["Image_ID"].tolist(),
        index=0,
    )
    selected_row = results_df.loc[results_df["Image_ID"] == selected_image].iloc[0]
    render_risk_card(
        sample_id=str(selected_row["Image_ID"]),
        score=float(selected_row["mviTIMES_Score"]),
        parent=str(selected_row["Representative_Parent"]),
    )

    st.markdown("#### Patient-level score table")
    formatted_df = results_df.copy()
    numeric_columns = [
        "mviTIMES_Score",
        "Parent_Score_Min",
        "Parent_Score_Mean",
        "Parent_Score_Max",
    ]
    for column in numeric_columns:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.4f}")
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download score table",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name=download_name,
        mime="text/csv",
        use_container_width=True,
    )

    render_notice(
        "<strong>Interpretation boundary.</strong> The underlying AUC869 workflow was identified during "
        "retrospective model development. The web adapter performs label-blind inference "
        "by scoring each uploaded Parent independently and retaining the highest score. Its output is "
        "an exploratory risk index, not an externally validated diagnostic probability."
    )


@st.cache_data(show_spinner=False)
def load_tiff_image(file_bytes) -> np.ndarray:
    image = tifffile.imread(file_bytes)
    if image.ndim == 3 and image.shape[2] < min(image.shape[0], image.shape[1]) and image.shape[2] <= 10:
        image = np.transpose(image, (2, 0, 1))
    elif image.ndim == 2:
        image = image[np.newaxis, :, :]
    return image


def segment_cells_dapi(dapi_channel: np.ndarray) -> np.ndarray:
    blurred = filters.gaussian(dapi_channel, sigma=1.5)
    threshold = filters.threshold_otsu(blurred)
    binary = morphology.remove_small_objects(blurred > threshold, min_size=20)
    binary = ndimage.binary_fill_holes(binary)
    distance = ndimage.distance_transform_edt(binary)
    coordinates = feature.peak_local_max(distance, footprint=np.ones((5, 5)), labels=binary)
    local_maxima = np.zeros(distance.shape, dtype=bool)
    local_maxima[tuple(coordinates.T)] = True
    markers, _ = ndimage.label(local_maxima)
    return segmentation.watershed(-distance, markers, mask=binary)


def quantify_and_phenotype(
    labels: np.ndarray,
    channels: dict[str, np.ndarray],
    image_id: str,
    parent_id: str,
) -> pd.DataFrame:
    properties = measure.regionprops(labels)
    thresholds: dict[str, float] = {}
    for name, image in channels.items():
        if image is not None and name != "DAPI":
            positive_pixels = image[image > 0]
            thresholds[name] = (
                float(filters.threshold_otsu(positive_pixels))
                if len(positive_pixels)
                else float(image.mean())
            )

    def is_positive(coords: np.ndarray, marker: str) -> bool:
        image = channels.get(marker)
        if image is None:
            return False
        mean_intensity = np.mean(image[coords[:, 0], coords[:, 1]])
        return bool(mean_intensity > thresholds.get(marker, 0))

    rows = []
    for cell in properties:
        centroid_y, centroid_x = cell.centroid
        coords = cell.coords
        cell_type = "Other"
        if is_positive(coords, "CD56.NK") and is_positive(coords, "CXCR4"):
            cell_type = "CXCR4.CD56.NK"
        elif is_positive(coords, "CD163") and is_positive(coords, "CXCL12"):
            cell_type = "CXCL12.CD163.M2"
        elif is_positive(coords, "CD56.NK"):
            cell_type = "CD56.NK"
        elif is_positive(coords, "GPC3.Tumor"):
            cell_type = "GPC3.Tumor"
        elif is_positive(coords, "CD34.Endothelial"):
            cell_type = "CD34.Endothelial"
        rows.append(
            {
                "Image": image_id,
                "Parent": parent_id,
                "x.axis": centroid_x,
                "y.axis": centroid_y,
                "CellType": cell_type,
            }
        )
    return pd.DataFrame(rows)


if "nav_menu" not in st.session_state:
    st.session_state.nav_menu = "Overview"


def switch_to_workspace() -> None:
    st.session_state.nav_menu = "Analyze"


with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=114)
    st.markdown('<div class="sidebar-brand">mviTIMES</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-copy">Spatial immune microenvironment risk explorer</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    nav_selection = st.radio(
        "Navigation",
        ["Overview", "Analyze", "Methodology"],
        label_visibility="collapsed",
        key="nav_menu",
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-copy" style="margin-bottom:.6rem;">MODEL PROFILE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">ExtraTrees classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">20 spatial features</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">50 μm interaction radius</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">Research-use platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-copy">© 2026 mviTIMES Research Platform<br>For exploratory use only</div>',
        unsafe_allow_html=True,
    )


if nav_selection == "Overview":
    st.markdown('<div class="eyebrow">Spatial Immuno-Oncology Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Decode the spatial immune context of microvascular invasion.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-subtitle">
            mviTIMES is a research-use spatial pathology platform that transforms multiplex
            immunostaining coordinates into an interpretable exploratory risk score for
            microvascular invasion (MVI).
        </div>
        <span class="pill">Single-cell coordinates</span>
        <span class="pill">NK–M2 spatial interactions</span>
        <span class="pill">ExtraTrees ensemble</span>
        <span class="pill">Research-use only</span>
        """,
        unsafe_allow_html=True,
    )

    section_heading(
        "From tissue coordinates to an exploratory risk index",
        "The workflow combines cell composition with local 50 μm spatial neighborhoods to summarize the immune architecture associated with MVI.",
    )
    steps = st.columns(4)
    with steps[0]:
        render_step("01", "Upload", "Import cell-level CSV coordinates or explore the TIFF demonstration workflow.")
    with steps[1]:
        render_step("02", "Quantify", "Measure densities, proportions, nearest-neighbor distances and local neighborhoods.")
    with steps[2]:
        render_step("03", "Model", "Apply a compact 20-feature ExtraTrees ensemble derived from the AUC869 workflow.")
    with steps[3]:
        render_step("04", "Review", "Inspect patient-level scores, representative regions and downloadable results.")

    section_heading("Core analytical layers")
    feature_cols = st.columns(3)
    with feature_cols[0]:
        render_feature_card(
            "Cellular composition",
            "Density and proportion features",
            "Quantifies CD34+ endothelial cells, CD56+ NK cells, CXCR4+ CD56+ NK cells, CXCL12+ CD163+ M2 cells and GPC3+ tumor cells.",
        )
    with feature_cols[1]:
        render_feature_card(
            "Spatial neighborhood",
            "NK–M2 interactions within 50 μm",
            "Captures local immune organization using directional nearest-neighbor and neighborhood-density measurements.",
        )
    with feature_cols[2]:
        render_feature_card(
            "Risk exploration",
            "Compact ExtraTrees ensemble",
            "Generates a reproducible research-use score while preserving clear boundaries around model interpretation.",
        )

    section_heading("Study context")
    context_cols = st.columns([1.05, 1])
    with context_cols[0]:
        st.markdown(
            """
            Microvascular invasion is an important histopathological indicator of tumor
            aggressiveness. Beyond conventional cell counting, multiplex imaging makes it
            possible to study how immune populations are arranged within tissue.

            mviTIMES focuses on the spatial relationship between `CXCR4+ CD56+ NK` cells and
            `CXCL12+ CD163+ M2` cells while integrating endothelial, NK-cell and tumor-context
            signals.
            """
        )
        render_notice(
            "<strong>Scope.</strong> This interface is intended for research exploration. "
            "It does not replace histopathological review, clinical judgment or "
            "independent validation."
        )
    with context_cols[1]:
        if TUTORIAL_PATH.exists():
            st.video(str(TUTORIAL_PATH))
        else:
            st.info("Tutorial video is not available in this deployment.")

    st.button("Open analysis workspace", type="primary", use_container_width=True, on_click=switch_to_workspace)


elif nav_selection == "Analyze":
    st.markdown('<div class="eyebrow">Analysis Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:3.35rem;">Generate an mviTIMES score.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-subtitle">
            Choose the coordinate workflow for routine scoring or the image workflow for
            an experimental end-to-end demonstration.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_notice(
        "<strong>Scoring strategy.</strong> Patient-level scores are generated from the highest-scoring "
        "uploaded Parent region. Uploaded data are processed within the active application session."
    )

    csv_tab, tiff_tab = st.tabs(["Coordinate CSV", "Multiplex TIFF demonstration"])

    with csv_tab:
        section_heading(
            "Coordinate-based scoring",
            "Recommended for cell-level outputs exported from established image-analysis pipelines such as HALO or QuPath.",
        )
        uploaded_csv = st.file_uploader(
            "Upload a cell-coordinate CSV",
            type=["csv"],
            key="coordinate_csv",
            help="Required columns: Image, Parent, x.axis, y.axis, CellType",
        )
        if uploaded_csv is not None:
            try:
                coordinate_df = load_csv_safely(uploaded_csv)
                validate_coordinate_csv(coordinate_df)
                summary_cols = st.columns(4)
                summary_cols[0].metric("Rows", f"{len(coordinate_df):,}")
                summary_cols[1].metric("Images", f"{coordinate_df['Image'].nunique():,}")
                summary_cols[2].metric("Parent regions", f"{coordinate_df[['Image', 'Parent']].drop_duplicates().shape[0]:,}")
                summary_cols[3].metric("Cell types", f"{coordinate_df['CellType'].nunique():,}")
                with st.expander("Preview uploaded coordinates"):
                    st.dataframe(coordinate_df.head(100), use_container_width=True, hide_index=True)

                if st.button("Run spatial risk analysis", type="primary", use_container_width=True, key="run_csv"):
                    if not MODEL_PATH.exists():
                        raise FileNotFoundError(f"Model package not found: {MODEL_PATH}")
                    with st.spinner("Extracting spatial features and applying the ExtraTrees ensemble..."):
                        predictor = load_predictor(str(MODEL_PATH))
                        score_df = predictor.predict_score(coordinate_df)
                    st.success("Spatial risk analysis completed.")
                    render_results(score_df)
            except Exception as error:
                st.error(f"Unable to score the uploaded CSV: {error}")

    with tiff_tab:
        section_heading(
            "Experimental multiplex-image workflow",
            "This demonstration performs DAPI-based segmentation, marker phenotyping and spatial scoring. Validate segmentation and phenotyping before any downstream interpretation.",
        )
        render_notice(
            "<strong>Experimental route.</strong> TIFF processing is included as a demonstration. "
            "For rigorous analyses, validate channel mapping, optical scale, segmentation quality "
            "and marker thresholds against your imaging platform."
        )
        setup_cols = st.columns(3)
        with setup_cols[0]:
            image_id = st.text_input("Sample ID", "Patient_001")
        with setup_cols[1]:
            parent_id = st.text_input("Region ID", "Core_Tumor")
        with setup_cols[2]:
            scale_factor = st.number_input("Optical scale (μm / px)", min_value=0.01, value=0.50, step=0.01)

        uploaded_tiff = st.file_uploader(
            "Upload a multiplex TIFF image",
            type=["tif", "tiff", "qptiff"],
            key="multiplex_tiff",
        )
        if uploaded_tiff is not None:
            try:
                with st.spinner("Reading multiplex image channels..."):
                    image_matrix = load_tiff_image(uploaded_tiff)
                channel_count = int(image_matrix.shape[0])
                st.success(
                    f"Loaded {channel_count} channels at {image_matrix.shape[2]} × {image_matrix.shape[1]} pixels."
                )

                marker_options = {
                    "Ignore channel": "IGNORE",
                    "DAPI | nuclei": "DAPI",
                    "GPC3 | tumor": "GPC3.Tumor",
                    "CD34 | endothelial": "CD34.Endothelial",
                    "CD56 | NK": "CD56.NK",
                    "CXCR4": "CXCR4",
                    "CD163": "CD163",
                    "CXCL12": "CXCL12",
                }
                section_heading("Channel mapping", "Assign a marker identity to each fluorescence channel.")
                channel_mapping: dict[int, str] = {}
                for batch_start in range(0, channel_count, 4):
                    channel_cols = st.columns(min(4, channel_count - batch_start))
                    for offset, column in enumerate(channel_cols):
                        channel_index = batch_start + offset
                        with column:
                            thumbnail = image_matrix[channel_index][::4, ::4]
                            lower, upper = np.percentile(thumbnail, (5, 99.5))
                            if upper == lower:
                                upper = lower + 1e-5
                            normalized = np.clip((thumbnail - lower) / (upper - lower), 0, 1)
                            figure, axis = plt.subplots(figsize=(2.5, 2.2))
                            axis.imshow(normalized, cmap="gray")
                            axis.axis("off")
                            st.pyplot(figure, use_container_width=True)
                            plt.close(figure)
                            marker_label = st.selectbox(
                                f"Channel {channel_index + 1}",
                                list(marker_options),
                                key=f"channel_{channel_index}",
                            )
                            channel_mapping[channel_index] = marker_options[marker_label]

                if st.button("Run TIFF demonstration analysis", type="primary", use_container_width=True):
                    mapped_markers = list(channel_mapping.values())
                    if "DAPI" not in mapped_markers:
                        raise ValueError("Assign one channel as DAPI before running segmentation.")
                    with st.status("Running image-to-score workflow...", expanded=True) as status:
                        channels = {
                            marker: image_matrix[index]
                            for index, marker in channel_mapping.items()
                            if marker != "IGNORE"
                        }
                        status.update(label="Segmenting DAPI-positive nuclei...", state="running")
                        labels = segment_cells_dapi(channels["DAPI"])
                        status.update(label="Assigning marker-defined cell phenotypes...", state="running")
                        spatial_df = quantify_and_phenotype(labels, channels, image_id, parent_id)
                        spatial_df["x.axis"] = spatial_df["x.axis"] * scale_factor
                        spatial_df["y.axis"] = spatial_df["y.axis"] * scale_factor
                        valid_df = spatial_df[spatial_df["CellType"] != "Other"].copy()
                        if valid_df.empty:
                            raise ValueError("No model-relevant cell phenotypes were detected.")
                        status.update(label="Applying spatial ExtraTrees risk model...", state="running")
                        predictor = load_predictor(str(MODEL_PATH))
                        score_df = predictor.predict_score(valid_df)
                        status.update(label="Analysis completed.", state="complete")

                    st.markdown("#### Extracted cell-coordinate matrix")
                    metadata_cols = st.columns(3)
                    metadata_cols[0].metric("Sample ID", image_id)
                    metadata_cols[1].metric("Model-relevant cells", f"{len(valid_df):,}")
                    metadata_cols[2].metric("Analysis time", datetime.now().strftime("%Y-%m-%d %H:%M"))
                    st.dataframe(valid_df.head(100), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download extracted cell coordinates",
                        data=valid_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{image_id}_cells.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    render_results(score_df, download_name=f"{image_id}_mviTIMES_score.csv")
            except Exception as error:
                st.error(f"Unable to complete TIFF demonstration analysis: {error}")


elif nav_selection == "Methodology":
    st.markdown('<div class="eyebrow">Methodology & Use Guidance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:3.35rem;">Transparent by design.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-subtitle">
            Review the input contract, model features and interpretation boundaries before
            using mviTIMES scores in a research workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Input schema")
    schema_df = pd.DataFrame(
        [
            ("Image", "Patient or specimen identifier", "Required"),
            ("Parent", "Tissue region identifier", "Required"),
            ("x.axis", "Cell x coordinate in μm", "Required"),
            ("y.axis", "Cell y coordinate in μm", "Required"),
            ("CellType", "Marker-defined cell phenotype", "Required"),
            ("MVI.State", "Known outcome label", "Not required for web inference"),
        ],
        columns=["Column", "Meaning", "Web inference"],
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    section_heading("Model-relevant cell phenotypes")
    phenotype_df = pd.DataFrame(
        [
            ("CD34.Endothelial", "CD34+", "Endothelial context"),
            ("CD56.NK", "CD56+", "NK-cell context"),
            ("CXCR4.CD56.NK", "CXCR4+ CD56+", "CXCR4-expressing NK-like phenotype"),
            ("CXCL12.CD163.M2", "CXCL12+ CD163+", "M2-like macrophage phenotype"),
            ("GPC3.Tumor", "GPC3+", "Tumor-cell context"),
        ],
        columns=["CellType", "Marker pattern", "Analytical role"],
    )
    st.dataframe(phenotype_df, use_container_width=True, hide_index=True)

    section_heading(
        "Model architecture",
        "The platform uses an ExtraTreesClassifier with 300 trees, maximum depth 3 and 20 fixed spatial features.",
    )
    method_cols = st.columns(3)
    with method_cols[0]:
        render_feature_card(
            "Composition",
            "Cell density and proportion",
            "Includes CXCR4+ CD56+ NK proportions, NK-cell composition and selected cell-density measurements.",
        )
    with method_cols[1]:
        render_feature_card(
            "Proximity",
            "Nearest-neighbor distance",
            "Measures directional distances between endothelial, NK-like and tumor-associated cellular populations.",
        )
    with method_cols[2]:
        render_feature_card(
            "Neighborhood",
            "50 μm local interaction density",
            "Captures the abundance and fraction of target cells in a biologically interpretable local radius.",
        )

    section_heading("Interpretation boundary")
    render_notice(
        "<strong>Model-development boundary.</strong> The AUC869 workflow used true MVI status "
        "during representative-Parent selection in training. The web adapter cannot reproduce that "
        "label-aware step for unknown patients. Instead, it scores all uploaded Parent regions and "
        "reports the maximum score. The displayed value is therefore an exploratory research index."
    )
    st.markdown(
        """
        The following claims should **not** be made from this platform alone:

        - The web score is a calibrated probability of MVI.
        - The development-stage AUC of `0.869` is an unbiased external-validation result.
        - A score above `0.5` establishes an MVI diagnosis.
        - The model can replace pathology review or clinical decision-making.
        """
    )

    section_heading("Frequently asked questions")
    with st.expander("What does the 0.5 cutoff mean?"):
        st.write(
            "The threshold is a default exploratory classification cutoff. Scores at or above 0.5 are labeled "
            "as higher exploratory risk. This cutoff is not a clinically validated intervention threshold."
        )
    with st.expander("Can I upload CSV data without MVI.State?"):
        st.write(
            "Yes. Web inference is label-blind. The required columns are Image, Parent, x.axis, y.axis and CellType."
        )
    with st.expander("Should I use the TIFF route for formal analyses?"):
        st.write(
            "Treat it as a demonstration unless segmentation, marker thresholds and phenotype assignments have "
            "been validated for your imaging platform. Coordinate CSV input from a quality-controlled pipeline is preferred."
        )
