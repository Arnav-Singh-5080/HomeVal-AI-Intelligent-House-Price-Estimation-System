"""
HomeVal AI - Intelligent House Price Estimation System
A production-ready Streamlit application for residential property price
prediction using a pre-trained Linear Regression pipeline.

Developer: Arnav Singh
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------

APP_NAME = "HomeVal AI"
APP_TAGLINE = "Intelligent House Price Estimation System"
APP_VERSION = "1.0.0"
COPYRIGHT_YEAR = datetime.now().year

MODEL_PATH = "model.pkl"
DATASET_PATH = "HousePricePrediction.csv"

DEVELOPER_NAME = "Arnav Singh"
DEVELOPER_EDUCATION = "Building Intelligent Solutions with Machine Learning"
DEVELOPER_COLLEGE = ""
DEVELOPER_GITHUB = "https://github.com/Arnav-Singh-5080"
DEVELOPER_LINKEDIN = "https://www.linkedin.com/in/arnav-singh-a87847351/"
DEVELOPER_EMAIL = "itsarnav.singh80@gmail.com"

FEATURE_COLUMNS = [
    "MSSubClass",
    "MSZoning",
    "LotArea",
    "LotConfig",
    "BldgType",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "Exterior1st",
    "BsmtFinSF2",
    "TotalBsmtSF",
]

TARGET_COLUMN = "SalePrice"

MS_SUBCLASS_OPTIONS = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]
MS_ZONING_OPTIONS = ["C (all)", "FV", "RH", "RL", "RM"]
LOT_CONFIG_OPTIONS = ["Corner", "CulDSac", "FR2", "FR3", "Inside"]
BLDG_TYPE_OPTIONS = ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"]
EXTERIOR_1ST_OPTIONS = [
    "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd",
    "HdBoard", "ImStucc", "MetalSd", "Plywood", "Stone", "Stucco",
    "VinylSd", "Wd Sdng", "WdShing",
]

MS_ZONING_LABELS = {
    "C (all)": "Commercial",
    "FV": "Floating Village Residential",
    "RH": "Residential High Density",
    "RL": "Residential Low Density",
    "RM": "Residential Medium Density",
}

BLDG_TYPE_LABELS = {
    "1Fam": "Single-family Detached",
    "2fmCon": "Two-family Conversion",
    "Duplex": "Duplex",
    "Twnhs": "Townhouse End Unit",
    "TwnhsE": "Townhouse Inside Unit",
}

LOT_CONFIG_LABELS = {
    "Corner": "Corner Lot",
    "CulDSac": "Cul-de-Sac",
    "FR2": "Frontage on 2 Sides",
    "FR3": "Frontage on 3 Sides",
    "Inside": "Inside Lot",
}


# ----------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="HomeVal AI | Intelligent House Price Estimation",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Inject the application-wide custom CSS theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

        :root {
            --color-primary: #1B4B8C;
            --color-primary-dark: #123561;
            --color-primary-light: #2E6BB8;
            --color-accent: #3A7CC4;
            --color-bg: #F5F7FA;
            --color-surface: #FFFFFF;
            --color-border: #E3E8EF;
            --color-text-dark: #1A2332;
            --color-text-muted: #5B6472;
            --color-text-light: #8A93A2;
            --radius-lg: 18px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --shadow-soft: 0 4px 20px rgba(27, 75, 140, 0.08);
            --shadow-medium: 0 8px 30px rgba(27, 75, 140, 0.12);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer, header[data-testid="stHeader"] {
            visibility: hidden;
            height: 0;
        }

        .stApp {
            background: linear-gradient(180deg, #F5F7FA 0%, #EEF2F7 100%);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        /* ---------- Hero ---------- */
        .hero-container {
            background: linear-gradient(135deg, var(--color-primary-dark) 0%, var(--color-primary) 55%, var(--color-accent) 100%);
            border-radius: var(--radius-lg);
            padding: 3rem 3rem 2.6rem 3rem;
            margin-bottom: 1.8rem;
            box-shadow: var(--shadow-medium);
            position: relative;
            overflow: hidden;
        }

        .hero-container::after {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0) 70%);
            border-radius: 50%;
        }

        .hero-eyebrow {
            display: inline-block;
            color: #DCE8FA;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-family: 'Manrope', sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            color: #FFFFFF;
            margin: 0 0 0.6rem 0;
            line-height: 1.15;
        }

        .hero-subtitle {
            color: #E3ECFB;
            font-size: 1.05rem;
            font-weight: 400;
            max-width: 720px;
            line-height: 1.6;
            margin: 0;
        }

        /* ---------- Section headers ---------- */
        .section-header {
            font-family: 'Manrope', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--color-text-dark);
            margin: 0.4rem 0 0.2rem 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .section-header::before {
            content: "";
            width: 5px;
            height: 22px;
            border-radius: 4px;
            background: linear-gradient(180deg, var(--color-primary) 0%, var(--color-accent) 100%);
            display: inline-block;
        }

        .section-subtext {
            color: var(--color-text-muted);
            font-size: 0.9rem;
            margin-bottom: 1.1rem;
        }

        /* ---------- Cards ---------- */
        .glass-card {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.6);
            border-radius: var(--radius-lg);
            padding: 1.6rem 1.8rem;
            box-shadow: var(--shadow-soft);
            margin-bottom: 1.3rem;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }

        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-medium);
        }

        .metric-card {
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            padding: 1.2rem 1.3rem;
            text-align: left;
            box-shadow: var(--shadow-soft);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-medium);
        }

        .metric-label {
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-family: 'Manrope', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--color-primary-dark);
        }

        /* ---------- Prediction result ---------- */
        .result-card {
            background: linear-gradient(135deg, var(--color-primary-dark) 0%, var(--color-primary) 60%, var(--color-accent) 100%);
            border-radius: var(--radius-lg);
            padding: 2.2rem 2rem;
            text-align: center;
            box-shadow: var(--shadow-medium);
            margin-top: 1rem;
        }

        .result-label {
            color: #DCE8FA;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        .result-value {
            font-family: 'Manrope', sans-serif;
            color: #FFFFFF;
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .result-note {
            color: #C9DAF4;
            font-size: 0.82rem;
            margin-top: 0.6rem;
        }

        /* ---------- Divider ---------- */
        .hv-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(27,75,140,0) 0%, var(--color-border) 20%, var(--color-border) 80%, rgba(27,75,140,0) 100%);
            margin: 1.8rem 0;
            border: none;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10233F 0%, #163A63 100%);
        }

        section[data-testid="stSidebar"] * {
            color: #E7EEF9 !important;
        }

        section[data-testid="stSidebar"] .sidebar-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: var(--radius-md);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        section[data-testid="stSidebar"] .sidebar-title {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #9FB6D9 !important;
            margin-bottom: 0.5rem;
        }

        section[data-testid="stSidebar"] .sidebar-name {
            font-family: 'Manrope', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }

        section[data-testid="stSidebar"] .sidebar-sub {
            font-size: 0.82rem;
            color: #B7C7E0 !important;
            line-height: 1.4;
        }

        section[data-testid="stSidebar"] a {
            text-decoration: none !important;
            color: #9FCBFF !important;
            font-size: 0.85rem;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.12);
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
            color: #FFFFFF;
            border: none;
            border-radius: var(--radius-sm);
            padding: 0.65rem 1.6rem;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
            box-shadow: 0 4px 14px rgba(27, 75, 140, 0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(27, 75, 140, 0.32);
            color: #FFFFFF;
        }

        .stButton > button:active {
            transform: translateY(0px);
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #FFFFFF;
            padding: 6px;
            border-radius: var(--radius-md);
            border: 1px solid var(--color-border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: var(--radius-sm);
            padding: 0.55rem 1.1rem;
            font-weight: 600;
            color: var(--color-text-muted);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%);
            color: #FFFFFF !important;
        }

        /* ---------- Inputs ---------- */
        div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
            border-radius: var(--radius-sm) !important;
            border: 1px solid var(--color-border) !important;
        }

        /* ---------- Footer ---------- */
        .hv-footer {
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-lg);
            padding: 1.8rem 2rem;
            margin-top: 2.5rem;
            text-align: center;
            box-shadow: var(--shadow-soft);
        }

        .hv-footer-title {
            font-family: 'Manrope', sans-serif;
            font-weight: 700;
            color: var(--color-text-dark);
            font-size: 1rem;
            margin-bottom: 0.3rem;
        }

        .hv-footer-links {
            color: var(--color-text-muted);
            font-size: 0.85rem;
            margin: 0.4rem 0;
        }

        .hv-footer-links a {
            color: var(--color-primary);
            text-decoration: none;
            font-weight: 600;
            margin: 0 0.5rem;
        }

        .hv-footer-copy {
            color: var(--color-text-light);
            font-size: 0.78rem;
            margin-top: 0.5rem;
        }

        /* ---------- Badge ---------- */
        .hv-badge {
            display: inline-block;
            background: rgba(27, 75, 140, 0.08);
            color: var(--color-primary-dark);
            border: 1px solid rgba(27, 75, 140, 0.18);
            border-radius: 999px;
            padding: 0.25rem 0.8rem;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }

        p, span, label, .stMarkdown {
            color: var(--color-text-dark);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load the trained model pipeline from disk."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    """Load the training dataset from disk, if available."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# UI HELPER COMPONENTS
# ----------------------------------------------------------------------------

def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-container">
            <span class="hero-eyebrow">Machine Learning &middot; Regression Analytics</span>
            <div class="hero-title">{APP_NAME}</div>
            <p class="hero-subtitle">
                {APP_TAGLINE}. HomeVal AI applies a supervised Linear Regression
                model, trained on historical residential property records, to
                estimate fair market sale prices from structural, locational,
                and construction attributes. Provide the property details
                below to generate an instant, data-driven valuation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> str:
    return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """


def render_dashboard_metrics(dataset: pd.DataFrame | None) -> None:
    st.markdown('<div class="section-header">Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Key facts about the underlying model and dataset.</div>', unsafe_allow_html=True)

    dataset_size = f"{len(dataset):,} records" if dataset is not None else "Not available"

    cols = st.columns(5)
    metrics = [
        ("Algorithm", "Linear Regression"),
        ("Problem Type", "Regression"),
        ("Input Features", str(len(FEATURE_COLUMNS))),
        ("Target Variable", TARGET_COLUMN),
        ("Dataset Size", dataset_size),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(render_metric_card(label, value), unsafe_allow_html=True)

    st.markdown('<hr class="hv-divider">', unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-title">Developer Profile</div>
                <div class="sidebar-name">{DEVELOPER_NAME}</div>
                <div class="sidebar-sub">{DEVELOPER_EDUCATION}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-title">About This Project</div>
                <div class="sidebar-sub"><b>Algorithm:</b> Linear Regression</div>
                <div class="sidebar-sub"><b>Problem Type:</b> Supervised Regression</div>
                <div class="sidebar-sub"><b>Dataset:</b> {DATASET_PATH}</div>
                <div class="sidebar-sub"><b>Features Used:</b> {len(FEATURE_COLUMNS)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-title" style="margin-top:0.4rem;">Navigation</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-sub" style="line-height:2;">
                Price Prediction<br>
                Dataset Explorer<br>
                Visual Analytics<br>
                Model Information<br>
                About Project
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-card" style="margin-top:1rem;">
                <div class="sidebar-title">Quick Links</div>
                <div class="sidebar-sub">
                    <a href="{DEVELOPER_GITHUB}" target="_blank">GitHub Profile</a><br><br>
                    <a href="{DEVELOPER_LINKEDIN}" target="_blank">LinkedIn Profile</a><br><br>
                    <a href="mailto:{DEVELOPER_EMAIL}">{DEVELOPER_EMAIL}</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-sub" style="text-align:center; margin-top:0.6rem; opacity:0.75;">
                {APP_NAME} &middot; Version {APP_VERSION}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="hv-footer">
            <div class="hv-footer-title">{APP_NAME} &mdash; {APP_TAGLINE}</div>
            <div class="hv-footer-links">
                Developed by {DEVELOPER_NAME} &nbsp;|&nbsp;
                <a href="{DEVELOPER_GITHUB}" target="_blank">GitHub</a>
                <a href="{DEVELOPER_LINKEDIN}" target="_blank">LinkedIn</a>
                <a href="mailto:{DEVELOPER_EMAIL}">Email</a>
            </div>
            <div class="hv-footer-copy">
                Version {APP_VERSION} &nbsp;&middot;&nbsp; Copyright {COPYRIGHT_YEAR} {DEVELOPER_NAME}. All rights reserved.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# PREDICTION TAB
# ----------------------------------------------------------------------------

def render_prediction_tab(model, dataset: pd.DataFrame | None) -> None:
    if model is None:
        st.error(
            f"Model file '{MODEL_PATH}' could not be located or loaded. "
            "Please place a valid model.pkl file in the project directory to enable predictions."
        )
        return

    st.markdown('<div class="section-header">Property Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">General classification and lot details of the property.</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            ms_subclass = st.selectbox(
                "Building Class (MSSubClass)",
                options=MS_SUBCLASS_OPTIONS,
                index=MS_SUBCLASS_OPTIONS.index(60) if 60 in MS_SUBCLASS_OPTIONS else 0,
                help="Identifies the type of dwelling involved in the sale.",
            )
        with c2:
            ms_zoning = st.selectbox(
                "Zoning Classification (MSZoning)",
                options=MS_ZONING_OPTIONS,
                format_func=lambda x: f"{x} - {MS_ZONING_LABELS.get(x, x)}",
                index=MS_ZONING_OPTIONS.index("RL"),
                help="The general zoning classification of the sale.",
            )
        with c3:
            lot_area = st.number_input(
                "Lot Area (sq. ft.)",
                min_value=500,
                max_value=250000,
                value=8500,
                step=50,
                help="Lot size in square feet.",
            )

    st.markdown('<div class="section-header">Location Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Lot configuration and building type.</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            lot_config = st.selectbox(
                "Lot Configuration (LotConfig)",
                options=LOT_CONFIG_OPTIONS,
                format_func=lambda x: f"{x} - {LOT_CONFIG_LABELS.get(x, x)}",
                index=LOT_CONFIG_OPTIONS.index("Inside"),
                help="Configuration of the lot.",
            )
        with c2:
            bldg_type = st.selectbox(
                "Building Type (BldgType)",
                options=BLDG_TYPE_OPTIONS,
                format_func=lambda x: f"{x} - {BLDG_TYPE_LABELS.get(x, x)}",
                index=BLDG_TYPE_OPTIONS.index("1Fam"),
                help="Type of dwelling.",
            )
        with c3:
            overall_cond = st.slider(
                "Overall Condition (OverallCond)",
                min_value=1,
                max_value=10,
                value=5,
                help="Rates the overall condition of the house (1 = Very Poor, 10 = Excellent).",
            )

    st.markdown('<div class="section-header">Construction Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Construction year, remodeling, and exterior material.</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)
        current_year = datetime.now().year
        with c1:
            year_built = st.number_input(
                "Year Built (YearBuilt)",
                min_value=1870,
                max_value=current_year,
                value=2003,
                step=1,
                help="Original construction date.",
            )
        with c2:
            year_remod = st.number_input(
                "Year Remodeled (YearRemodAdd)",
                min_value=1870,
                max_value=current_year,
                value=2003,
                step=1,
                help="Remodel date (same as construction date if no remodeling or additions).",
            )
        with c3:
            exterior_1st = st.selectbox(
                "Exterior Covering (Exterior1st)",
                options=EXTERIOR_1ST_OPTIONS,
                index=EXTERIOR_1ST_OPTIONS.index("VinylSd"),
                help="Exterior covering on the house.",
            )

    st.markdown('<div class="section-header">Basement Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Basement finished area and total basement area.</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            bsmt_fin_sf2 = st.number_input(
                "Basement Finished Area — Type 2 (BsmtFinSF2, sq. ft.)",
                min_value=0,
                max_value=5000,
                value=0,
                step=10,
                help="Type 2 finished square feet in the basement.",
            )
        with c2:
            total_bsmt_sf = st.number_input(
                "Total Basement Area (TotalBsmtSF, sq. ft.)",
                min_value=0,
                max_value=10000,
                value=900,
                step=10,
                help="Total square feet of basement area.",
            )

    st.markdown('<hr class="hv-divider">', unsafe_allow_html=True)

    input_valid = True
    if year_remod < year_built:
        st.warning("Year Remodeled cannot be earlier than Year Built. Please review the construction details.")
        input_valid = False

    predict_col = st.columns([1, 1, 1])[1]
    with predict_col:
        predict_clicked = st.button("Predict House Price", use_container_width=True)

    if predict_clicked:
        if not input_valid:
            st.error("Please correct the highlighted input before generating a prediction.")
            return

        with st.spinner("Analyzing property attributes and generating estimate..."):
            input_df = pd.DataFrame(
                [{
                    "MSSubClass": ms_subclass,
                    "MSZoning": ms_zoning,
                    "LotArea": lot_area,
                    "LotConfig": lot_config,
                    "BldgType": bldg_type,
                    "OverallCond": overall_cond,
                    "YearBuilt": year_built,
                    "YearRemodAdd": year_remod,
                    "Exterior1st": exterior_1st,
                    "BsmtFinSF2": bsmt_fin_sf2,
                    "TotalBsmtSF": total_bsmt_sf,
                }],
                columns=FEATURE_COLUMNS,
            )

            try:
                prediction = model.predict(input_df)[0]
                prediction = max(float(prediction), 0.0)
            except Exception as exc:
                st.error(f"An error occurred while generating the prediction: {exc}")
                return

        formatted_price = f"${prediction:,.2f}"
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Property Value</div>
                <div class="result-value">{formatted_price}</div>
                <div class="result-note">Generated using a trained Linear Regression pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success("Prediction generated successfully.")


# ----------------------------------------------------------------------------
# DATASET TAB
# ----------------------------------------------------------------------------

def render_dataset_tab(dataset: pd.DataFrame | None) -> None:
    if dataset is None:
        st.error(
            f"Dataset file '{DATASET_PATH}' could not be located. "
            "Please place the dataset file in the project directory to view dataset insights."
        )
        return

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Explore the structure and quality of the training dataset.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_metric_card("Total Records", f"{dataset.shape[0]:,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Total Columns", f"{dataset.shape[1]:,}"), unsafe_allow_html=True)
    with c3:
        missing_total = int(dataset.isnull().sum().sum())
        st.markdown(render_metric_card("Missing Values", f"{missing_total:,}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(dataset.head(10), use_container_width=True)

    with st.expander("Column Names and Data Types"):
        dtype_df = pd.DataFrame({
            "Column": dataset.columns,
            "Data Type": dataset.dtypes.astype(str).values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with st.expander("Missing Values by Column"):
        missing_df = dataset.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df = missing_df[missing_df["Missing Values"] > 0].sort_values(
            "Missing Values", ascending=False
        )
        if missing_df.empty:
            st.info("No missing values detected in the dataset.")
        else:
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

    with st.expander("Summary Statistics"):
        st.dataframe(dataset.describe().transpose(), use_container_width=True)


# ----------------------------------------------------------------------------
# VISUALIZATION TAB
# ----------------------------------------------------------------------------

def apply_plot_theme() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E3E8EF",
        "axes.labelcolor": "#1A2332",
        "text.color": "#1A2332",
        "xtick.color": "#5B6472",
        "ytick.color": "#5B6472",
        "grid.color": "#E3E8EF",
        "font.family": "sans-serif",
    })


def render_visualization_tab(dataset: pd.DataFrame | None) -> None:
    if dataset is None:
        st.error(
            f"Dataset file '{DATASET_PATH}' could not be located. "
            "Please place the dataset file in the project directory to view visual analytics."
        )
        return

    apply_plot_theme()
    st.markdown('<div class="section-header">Visual Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Distribution and relationship insights derived from the dataset.</div>', unsafe_allow_html=True)

    if TARGET_COLUMN in dataset.columns:
        st.markdown("**Sale Price Distribution**")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        sns.histplot(dataset[TARGET_COLUMN].dropna(), bins=40, color="#1B4B8C", kde=True, ax=ax)
        ax.set_xlabel("Sale Price")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    numeric_df = dataset.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(9, 6))
        corr = numeric_df.corr()
        sns.heatmap(
            corr,
            cmap="Blues",
            linewidths=0.4,
            linecolor="#F5F7FA",
            ax=ax,
            cbar_kws={"shrink": 0.75},
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**Feature Distribution**")
    numeric_columns = [c for c in numeric_df.columns if c != TARGET_COLUMN]
    if numeric_columns:
        selected_feature = st.selectbox("Select a feature to visualize", options=numeric_columns)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        sns.histplot(dataset[selected_feature].dropna(), bins=35, color="#3A7CC4", kde=True, ax=ax)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No additional numeric features available for distribution analysis.")


# ----------------------------------------------------------------------------
# MODEL INFORMATION TAB
# ----------------------------------------------------------------------------

def render_model_info_tab(model) -> None:
    st.markdown('<div class="section-header">Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Technical details of the trained prediction pipeline.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Algorithm**")
        st.write("Linear Regression")
        st.markdown("**Regression Type**")
        st.write("Multiple Linear Regression (Ordinary Least Squares)")
        st.markdown("**Target Variable**")
        st.write(TARGET_COLUMN)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Categorical Encoding**")
        st.write("One-Hot Encoding (unknown categories safely ignored at inference time)")
        st.markdown("**Numerical Features**")
        st.write("Passed through without transformation")
        st.markdown("**Pipeline Stages**")
        st.write("Preprocessing (Column Transformer) followed by Linear Regression estimator")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Prediction Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        1. User submits property attributes through the input form.<br>
        2. Inputs are assembled into a single-row structured DataFrame.<br>
        3. Categorical fields are transformed using one-hot encoding.<br>
        4. Numerical fields are passed through unchanged.<br>
        5. The combined feature vector is passed to the trained Linear Regression model.<br>
        6. The model returns a continuous estimated sale price, formatted as currency.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is not None:
        st.markdown('<div class="section-header" style="margin-top:1rem;">Input Feature Schema</div>', unsafe_allow_html=True)
        schema_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Description": [
                "Type of dwelling involved in the sale",
                "General zoning classification",
                "Lot size in square feet",
                "Configuration of the lot",
                "Type of dwelling",
                "Overall condition rating (1-10)",
                "Original construction year",
                "Remodel or addition year",
                "Exterior covering material",
                "Type 2 finished basement area (sq. ft.)",
                "Total basement area (sq. ft.)",
            ],
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Model file '{MODEL_PATH}' is not currently loaded.")


# ----------------------------------------------------------------------------
# ABOUT TAB
# ----------------------------------------------------------------------------

def render_about_tab() -> None:
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="glass-card">
            <b>Project Goal</b><br>
            {APP_NAME} was built to demonstrate an end-to-end supervised
            machine learning workflow: from raw tabular housing data to a
            deployed, interactive prediction interface. The system estimates
            residential sale prices from structural, locational, and
            construction attributes using a Linear Regression model.
        </div>

        <div class="glass-card">
            <b>Workflow</b><br>
            Data collection and cleaning, exploratory analysis, feature
            preprocessing with one-hot encoding for categorical variables,
            model training with Linear Regression, model serialization with
            joblib, and finally deployment through this Streamlit interface.
        </div>

        <div class="glass-card">
            <b>Machine Learning Pipeline</b><br>
            The trained pipeline consists of a Column Transformer that
            one-hot encodes categorical fields (zoning, lot configuration,
            building type, and exterior covering) while passing numerical
            fields through unchanged, followed by a Linear Regression
            estimator that outputs the predicted sale price.
        </div>

        <div class="glass-card">
            <b>Dataset</b><br>
            The model was trained on {DATASET_PATH}, a residential property
            dataset containing structural, locational, and construction
            attributes alongside historical sale prices.
        </div>

        <div class="glass-card">
            <b>Why Linear Regression</b><br>
            Linear Regression offers a transparent, interpretable baseline
            for price estimation, establishing a clear relationship between
            property attributes and sale price while remaining computationally
            efficient and easy to validate.
        </div>

        <div class="glass-card">
            <b>Future Improvements</b><br>
            Planned enhancements include ensemble models such as Random
            Forest and Gradient Boosting for improved accuracy, expanded
            feature engineering, cross-validation based model selection,
            and confidence intervals around predictions.
        </div>

        <div class="glass-card">
            <b>Developer Information</b><br>
            {DEVELOPER_NAME}<br>
            {DEVELOPER_EDUCATION}<br><br>
            <a href="{DEVELOPER_GITHUB}" target="_blank">GitHub</a> &nbsp;|&nbsp;
            <a href="{DEVELOPER_LINKEDIN}" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
            <a href="mailto:{DEVELOPER_EMAIL}">{DEVELOPER_EMAIL}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------------

def main() -> None:
    inject_custom_css()
    render_sidebar()
    render_hero()

    model = load_model(MODEL_PATH)
    dataset = load_dataset(DATASET_PATH)

    render_dashboard_metrics(dataset)

    tabs = st.tabs([
        "Price Prediction",
        "Dataset Explorer",
        "Visual Analytics",
        "Model Information",
        "About Project",
    ])

    with tabs[0]:
        render_prediction_tab(model, dataset)

    with tabs[1]:
        render_dataset_tab(dataset)

    with tabs[2]:
        render_visualization_tab(dataset)

    with tabs[3]:
        render_model_info_tab(model)

    with tabs[4]:
        render_about_tab()

    render_footer()


if __name__ == "__main__":
    main()