"""
Streamlit frontend demo for CBM NLP API service.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import altair as alt
import json
import os
import glob
import ast
import base64
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from typing import List


# Page configuration
st.set_page_config(
    page_title="EssayCBM",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded"
)

# Available modes
AVAILABLE_MODES = ["joint"]

def get_available_models_from_filesystem() -> list:
    """Dynamically get available models from saved_models/original directory."""
    models_dir = "/Users/scott/repos/CBM/saved_models/original"
    if not os.path.exists(models_dir):
        return ["bert-base-uncased", "roberta-base"]  # fallback
    
    # Get all directories in the models folder
    model_dirs = [d for d in os.listdir(models_dir) 
                  if os.path.isdir(os.path.join(models_dir, d))]
    
    # Filter out hidden directories and sort
    model_dirs = [d for d in model_dirs if not d.startswith('.')]
    model_dirs.sort()
    
    return model_dirs if model_dirs else ["bert-base-uncased", "roberta-base"]  # fallback

# Get available models dynamically
AVAILABLE_MODELS = get_available_models_from_filesystem()

# Concept name mapping
CONCEPT_FULL_NAMES = {
    # QA Dataset concepts
    "FC": "Focus/Clarity",
    "CC": "Coherence/Cohesion", 
    "TU": "Task Understanding",
    "CP": "Critical Thinking",
    "R": "Relevance",
    "DU": "Depth/Understanding",
    "EE": "Evidence/Examples",
    "FR": "Flow/Readability",
    
    # Essay Dataset concepts
    "TC": "Task Completion",
    "UE": "Understanding/Explanation",
    "OC": "Organization/Clarity",
    "GM": "Grammar/Mechanics",
    "VA": "Vocabulary/Accuracy",
    "SV": "Support/Validation",
    "CTD": "Critical Thinking/Depth",
    "FR": "Flow/Readability",
    
    # CEBaB Dataset concepts
    "Food": "Food Quality",
    "Ambiance": "Ambiance/Atmosphere",
    "Service": "Service Quality",
    "Noise": "Noise Level",
    "cleanliness": "Cleanliness",
    "price": "Price/Value",
    "location": "Location",
    "menu_variety": "Menu Variety",
    "waiting_time": "Waiting Time",
    "waiting_area": "Waiting Area",
    
    # IMDB Dataset concepts
    "acting": "Acting Performance",
    "storyline": "Storyline/Plot",
    "emotional": "Emotional Impact",
    "cinematography": "Cinematography",
    "soundtrack": "Soundtrack/Music",
    "directing": "Directing",
    "background": "Background Setting",
    "editing": "Editing"
}


# Utilities for results discovery and loading
# Primary and fallback results directories
RESULTS_DIR_PRIMARY = "/Users/scott/repos/CBM_NLP/cbm/results"
try:
    RESULTS_DIR_FALLBACK = str((Path(__file__).resolve().parent.parent / "cbm" / "results"))
except Exception:
    RESULTS_DIR_FALLBACK = str((Path.cwd() / "cbm" / "results"))

KNOWN_DATASETS = ["essay", "imdb", "cebab", "qa"]


@st.cache_data(show_spinner=False)
def get_results_dir() -> str:
    return RESULTS_DIR_PRIMARY if os.path.isdir(RESULTS_DIR_PRIMARY) else RESULTS_DIR_FALLBACK


@st.cache_data(show_spinner=False)
def list_result_csvs(dir_path: str) -> list:
    files = glob.glob(os.path.join(dir_path, "*.csv"))
    files = [f for f in files if os.path.basename(f).lower() != "table1.csv"]
    files.sort(key=os.path.getmtime, reverse=True)
    return files


@st.cache_data(show_spinner=False)
def infer_dataset_from_file(path: str) -> str:
    try:
        df_head = pd.read_csv(path, nrows=1)
        if "dataset" in df_head.columns and pd.notna(df_head.loc[0, "dataset"]):
            return str(df_head.loc[0, "dataset"]).strip().lower()
    except Exception:
        pass
    name = os.path.basename(path).lower()
    for ds in KNOWN_DATASETS:
        if ds in name:
            return ds
    return "unknown"


@st.cache_data(show_spinner=False)
def group_files_by_dataset(files: list) -> Dict[str, list]:
    groups: Dict[str, list] = {}
    for f in files:
        ds = infer_dataset_from_file(f)
        groups.setdefault(ds, []).append(f)
    for ds in groups:
        groups[ds].sort(key=os.path.getmtime, reverse=True)
    return groups


@st.cache_data(show_spinner=False)
def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["score", "concept_score"]:
        if col in df.columns:
            try:
                parsed = df[col].apply(
                    lambda s: (ast.literal_eval(s)[0] if isinstance(s, str) and s.startswith("[") else None)
                )
                try:
                    has_any = parsed.notna().any()
                except Exception:
                    has_any = False
                if has_any:
                    # Use explicit, user-friendly metric names
                    names = [
                        'Task Accuracy', 'Task Macro-F1'
                    ] if col == 'score' else [
                        'Concept Accuracy', 'Concept Macro-F1'
                    ]
                    out = pd.DataFrame(parsed.tolist(), columns=names)
                    df = pd.concat([df, out], axis=1)
            except Exception:
                pass
    return df


def render_model_performance_section() -> None:
    dir_path = get_results_dir()
    files = list_result_csvs(dir_path)
    if not files:
        st.info("No result CSVs found.")
        return
    groups = group_files_by_dataset(files)
    try:
        newest_ds = max(groups.items(), key=lambda kv: os.path.getmtime(kv[1][0]))[0]
    except Exception:
        newest_ds = sorted(groups.keys())[0]
    ds_names = sorted(groups.keys(), key=lambda k: (k != newest_ds, k))
    selected_ds = st.selectbox("Select dataset", ds_names, index=0)
    ds_files = groups[selected_ds]
    file_labels = [os.path.basename(f) for f in ds_files]
    selected_label = st.selectbox("Select results file", file_labels, index=0)
    selected_path = ds_files[file_labels.index(selected_label)]
    df_results = load_results_csv(selected_path)

    if not df_results.empty:
        summary = df_results.iloc[0]
        metric_defs = [
            ("Task Accuracy", "Task Accuracy"),
            ("Task Macro-F1", "Task Macro-F1"),
            ("Concept Accuracy", "Concept Accuracy"),
            ("Concept Macro-F1", "Concept Macro-F1"),
        ]
        available = [(label, col) for (label, col) in metric_defs if col in df_results.columns]
        if available:
            cols = st.columns(len(available))
            for idx, (label, col) in enumerate(available):
                try:
                    val = float(summary[col])
                    cols[idx].metric(label, f"{val:.3f}")
                except Exception:
                    cols[idx].metric(label, "-")

    display_cols = [
        c for c in [
            'dataset', 'data_type', 'function', 'model',
            'Task Accuracy', 'Task Macro-F1', 'Concept Accuracy', 'Concept Macro-F1'
        ] if c in df_results.columns
    ]
    st.dataframe(df_results[display_cols] if display_cols else df_results, use_container_width=True)


def check_backend_connection(base_url: str) -> Dict[str, Any]:
    """Check if backend is accessible and get status."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "connected", "data": response.json()}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_available_models(base_url: str) -> Optional[Dict[str, Any]]:
    """Get available models from backend."""
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def api_register(base_url: str, username: str, password: str) -> bool:
    try:
        resp = requests.post(f"{base_url}/register", json={"username": username, "password": password}, timeout=10)
        if resp.status_code == 200:
            return True
        if resp.status_code == 409:
            st.warning("Username already exists.")
            return False
        st.error(f"Register failed: {resp.text}")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return False


def api_login(base_url: str, username: str, password: str) -> bool:
    try:
        resp = requests.post(f"{base_url}/login", json={"username": username, "password": password}, timeout=10)
        if resp.status_code == 200 and resp.json().get("ok"):
            return True
        st.error("Invalid credentials.")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return False


def api_save_grade(base_url: str, payload: Dict[str, Any]) -> Optional[int]:
    try:
        resp = requests.post(f"{base_url}/grade_history/save", json=payload, timeout=20)
        if resp.status_code == 200:
            return int(resp.json().get("id"))
        st.error(f"Save history failed: {resp.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def api_list_history(base_url: str, username: str, limit: int = 20) -> list:
    try:
        resp = requests.get(f"{base_url}/grade_history/list", params={"username": username, "limit": limit}, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("items", [])
        st.error(f"Load history failed: {resp.text}")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return []


def api_get_history_detail(base_url: str, username: str, record_id: int) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{base_url}/grade_history/detail", params={"username": username, "id": record_id}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Load detail failed: {resp.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def api_delete_history(base_url: str, username: str, record_id: int) -> bool:
    try:
        resp = requests.post(f"{base_url}/grade_history/delete", json={"username": username, "id": record_id}, timeout=10)
        if resp.status_code == 200:
            return True
        st.error(f"Delete failed: {resp.text}")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return False


def predict_single_text(base_url: str, text: str, model_name: str, mode: str) -> Optional[Dict[str, Any]]:
    """Send prediction request to backend."""
    try:
        payload = {
            "text": text,
            "model_name": model_name,
            "mode": mode
        }
        response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None




def format_prediction_icon(prediction: int, num_classes: int) -> str:
    """Format prediction icon based on number of classes."""
    return ""


def display_rating_highlight(rating: int, num_classes: int, confidence: float):
    """Display rating as large prominent text with confidence."""
    max_rating = num_classes
    confidence_pct = confidence * 100
    
    # Compact, centered card layout with reduced height
    st.markdown(
        f"""
        <div style="min-height:150px; padding:12px 8px; display:flex; align-items:center; justify-content:center;">
          <div style="text-align:center; max-width:520px; color:#111;">
            <div style="display:flex; align-items:center; justify-content:center; gap:8px; margin-bottom:6px;">
              <div style="font-size:20px; font-weight:700;">Rating: {rating}/{max_rating}</div>
              <div style="font-size:24px;">{"⭐" * rating if num_classes == 6 else ""}</div>
            </div>
            <div style="font-size:14px; color:#555;">
              <strong>Confidence:</strong> {confidence_pct:.1f}% &nbsp; | &nbsp; <strong>Max Score:</strong> {max_rating}
            </div>
          </div> 
        </div>
        """,
        unsafe_allow_html=True,
    )

def display_rating_compact(rating: int, num_classes: int, confidence: float):
    """Compact, left-aligned rating block for side placement."""
    max_rating = num_classes
    confidence_pct = confidence * 100
    st.markdown(
        f"""
        <div style="padding:4px 0; text-align:center;">
          <div style="font-size:24px; font-weight:800; margin-bottom:8px; color:#111;">
            Rating: {rating}/{max_rating}
          </div>
          <div style="font-size:28px; margin-bottom:10px;">
            {"⭐" * rating if num_classes == 6 else ""}
          </div>
          <div style="font-size:16px; color:#555;">
            <strong>Confidence:</strong> {confidence_pct:.1f}% &nbsp; | &nbsp; <strong>Max Score:</strong> {max_rating}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def display_probability_chart(probabilities: list, rating: int, num_classes: int, confidence: float):
    """Create 2:8 column layout with rating info on left and horizontal bar chart on right."""
    st.markdown("### Probability Distribution")
    
    # Create 2:8 column layout
    col1, col2 = st.columns([2, 8])
    
    with col1:
        st.markdown("**Rating Summary**")
        st.metric("Assigned Rating", f"{rating}/{num_classes}")
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.metric("Max Score", num_classes)
        
        # Show highest probability info
        max_prob_idx = probabilities.index(max(probabilities))
        st.markdown(f"**Highest:** Score {max_prob_idx + 1}")
        st.markdown(f"**Probability:** {max(probabilities)*100:.1f}%")
    
    with col2:
        # Create Plotly bar chart with tooltip and always-visible labels above bars
        # Ensure all classes 1..num_classes are present on the x-axis
        num_classes = len(probabilities)
        all_scores = [str(i + 1) for i in range(num_classes)]
        df = pd.DataFrame({"Score": all_scores})
        df["Probability"] = df["Score"].apply(
            lambda s: float(probabilities[int(s) - 1]) if int(s) - 1 < len(probabilities) else 0.0
        )
        fig = px.bar(df, x="Score", y="Probability")
        fig.update_traces(
            texttemplate="%{y:.1%}",
            textposition="outside",
            hovertemplate="Score: %{x}<br>Probability: %{y:.1%}<extra></extra>"
        )
        fig.update_yaxes(range=[0, 1], title_text="Probability")
        fig.update_xaxes(title_text="Score", categoryorder="array", categoryarray=all_scores)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), uniformtext_minsize=10, uniformtext_mode="hide")
        st.plotly_chart(fig, use_container_width=True)


def display_concept_cards(concept_predictions: list, show_header: bool = True):
    """Display concept cards with 2 cards per row."""
    if not concept_predictions:
        return
    if show_header:
        st.markdown('<div style="text-align:center;"><h2>Concept Analysis</h2></div>', unsafe_allow_html=True)
    
    # Color mapping for icons - handle different sentiment labels dynamically
    def get_icon_for_sentiment(sentiment: str) -> str:
        """Get appropriate icon based on sentiment label."""
        sentiment_lower = sentiment.lower()
        
        # Handle numeric labels (1-5 scale)
        if sentiment.isdigit():
            score = int(sentiment)
            if score <= 2:
                return "🔴"
            elif score == 3:
                return "🟡"
            else:
                return "🟢"
        
        # Handle text labels
        if any(word in sentiment_lower for word in ['negative', 'low', 'very low']):
            return "🔴"
        elif any(word in sentiment_lower for word in ['neutral', 'medium']):
            return "🟡"
        elif any(word in sentiment_lower for word in ['positive', 'high', 'very high']):
            return "🟢"
        else:
            return ""
    
    # Display cards with 2 per row
    per_row = 2
    for start in range(0, len(concept_predictions), per_row):
        cols = st.columns(per_row)
        for j in range(per_row):
            concept_idx = start + j
            if concept_idx < len(concept_predictions):
                concept = concept_predictions[concept_idx]
                with cols[j]:
                    # Get concept info
                    concept_name = concept['concept_name']
                    full_name = CONCEPT_FULL_NAMES.get(concept_name, concept_name)
                    prediction = concept['prediction']
                    probs = concept['probabilities']
                    
                    # Get styling
                    icon = get_icon_for_sentiment(prediction)
                    
                    # Get top probability info
                    top_prob = max(probs.values())
                    top_sentiment = max(probs, key=probs.get)
                    
                    # Create simple card without borders
                    card_html = f"""
                    <div style="
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 5px;
                        background-color: #f8f9fa;
                    ">
                        <div style="text-align:center; margin: 0 0 2px 0; color: #333;">
                            <span><strong>{icon} {full_name}</strong></span>
                        </div>
                        <div style="text-align:center; color: #888;">
                            <span><strong>Grading:</strong> {prediction}</span>
                            &nbsp; | &nbsp;
                            <span><strong>Top:</strong> {top_sentiment} ({top_prob*100:.1f}%)</span>
                        </div>
                    </div>
                    """
                    
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Add bar chart with in-bar labels inside the card
                    # Build dataframe; if labels are numeric 1..5, ensure all five categories appear
                    sentiment_labels = list(probs.keys())
                    are_numeric = all(str(k).isdigit() for k in sentiment_labels)
                    if are_numeric:
                        full_labels = [str(i) for i in range(1, 6)]
                        concept_df = pd.DataFrame({
                            "Sentiment": full_labels,
                            "Probability": [float(probs.get(lbl, 0.0)) for lbl in full_labels]
                        })
                    else:
                        concept_df = pd.DataFrame({
                            "Sentiment": sentiment_labels,
                            "Probability": [probs[k] for k in sentiment_labels]
                        })
                    fig = px.bar(concept_df, x="Sentiment", y="Probability")
                    fig.update_traces(
                        texttemplate="%{y:.1%}",
                        textposition="outside",
                        hovertemplate="Sentiment: %{x}<br>Probability: %{y:.1%}<extra></extra>"
                    )
                    fig.update_yaxes(range=[0, 1], title_text=None)
                    if are_numeric:
                        fig.update_xaxes(title_text=None, categoryorder="array", categoryarray=[str(i) for i in range(1, 6)])
                    else:
                        fig.update_xaxes(title_text=None)
                    fig.update_layout(height=300, uniformtext_minsize=9, uniformtext_mode="hide")
                    st.plotly_chart(fig, use_container_width=True)
        # Add a subtle divider between rows
        if start + per_row < len(concept_predictions):
            st.markdown(
                "<hr style='border:0; border-top:1px solid #b5b5b5; margin: 6px 0;'/>",
                unsafe_allow_html=True,
            )


def main():
    
    # Moved model performance section under Predict button in Tab 1
    # Backend config (sidebar removed)
    # Restore username from URL query params if present
    if "username" not in st.session_state:
        restored_user = None
        try:
            qp = st.query_params  # new API
            restored_user = qp.get("u")
            if isinstance(restored_user, list):
                restored_user = restored_user[0] if restored_user else None
        except Exception:
            try:
                qp = st.experimental_get_query_params()  # fallback API
                ulist: Optional[List[str]] = qp.get("u")
                restored_user = ulist[0] if ulist else None
            except Exception:
                restored_user = None
        if restored_user:
            st.session_state["username"] = restored_user
    
    backend_url = "http://localhost:8000"
    connection_status = check_backend_connection(backend_url)
    # Columns-based layout (avoid raw HTML wrappers to keep contents inside columns)
    if connection_status["status"] != "connected":
        st.warning("⚠️ Please ensure the backend service is running and accessible.")
        return
    # Reduce default top padding so content sits higher on the page (apply after login)
    if "username" in st.session_state:
        st.markdown(
            """
            <style>
              [data-testid="stAppViewContainer"] .main .block-container{
                padding-top: 0.5rem;
                width: 60%;
                margin-left: auto;
                margin-right: auto;
              }
              /* Increase textarea font size */
              [data-testid="stTextArea"] textarea {
                font-size: 18px !important;
                line-height: 1.5 !important;
              }
              /* Center images rendered via st.image */
              .stImage img {
                display: block;
                margin-left: auto;
                margin-right: auto;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Authentication gate
    if "username" not in st.session_state:
        # Native layout: center the login box to half width using columns
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            with st.container(border=True):
                # Logo on login screen
                try:
                    logo_path = str((Path(__file__).resolve().parent / "assets" / "logo.png"))
                    if os.path.exists(logo_path):
                        with open(logo_path, "rb") as f:
                            b64_logo = base64.b64encode(f.read()).decode()
                        st.markdown(
                            f'<div style="text-align:center;"><img src="data:image/png;base64,{b64_logo}" width="160"/></div>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass
                st.markdown('<div style="text-align:center;"><h2>Login to Essay CBM</h2></div>', unsafe_allow_html=True)
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Login", use_container_width=True):
                        if u and p and api_login(backend_url, u, p):
                            st.session_state["username"] = u
                            # Persist username in URL query params
                            try:
                                # New API (Streamlit >=1.32)
                                st.query_params["u"] = u
                            except Exception:
                                # Fallback API
                                try:
                                    st.experimental_set_query_params(u=u)
                                except Exception:
                                    pass
                            st.rerun()
                with col_b:
                    if st.button("Register", use_container_width=True):
                        if u and p and api_register(backend_url, u, p):
                            st.success("Registration successful. Please login.")
        return

    # Top bar: logo + user info (no border)
    with st.container():
        try:
            logo_path = str((Path(__file__).resolve().parent / "assets" / "logo.png"))
            if os.path.exists(logo_path):
                with open(logo_path, "rb") as f:
                    b64_logo = base64.b64encode(f.read()).decode()
                # st.markdown(
                #     f'<div style="text-align:center;"><img src="data:image/png;base64,{b64_logo}" width="160"/></div>',
                #     unsafe_allow_html=True,
                # )
        except Exception:
            pass
        # Sidebar: title + welcome + logout + history
        st.sidebar.markdown(
            f'# 👋 Welcome back! {st.session_state["username"].title()}',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            # Clear username from URL query params
            try:
                # New API
                st.query_params.clear()
            except Exception:
                # Fallback clears all params
                st.experimental_set_query_params()
            st.rerun()
        st.sidebar.markdown("### History")
        if "history_summaries" not in st.session_state:
            st.session_state["history_summaries"] = api_list_history(backend_url, st.session_state["username"])
        # Refresh control
        # ctrl_col1, ctrl_col2 = st.sidebar.columns(2)
        # with ctrl_col1:
        #     if st.sidebar.button("Refresh history"):
        #         st.session_state["history_summaries"] = api_list_history(backend_url, st.session_state["username"])
        # No edit toggle; delete buttons always visible
        # Left-align sidebar buttons (history rows)
        st.markdown(
            """
            <style>
              [data-testid="stSidebar"] .stButton > button {
                justify-content: flex-start;
                text-align: left;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                width: 100%;
                max-width: 100%;
              }
              /* Narrower sidebar width to fit current content */
              section[data-testid="stSidebar"] {
                min-width: 320px !important;
                max-width: 350px !important;
              }
              section[data-testid="stSidebar"] > div:first-child {
                width: 100% !important;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )
        summaries = st.session_state.get("history_summaries", [])
        if summaries:
            # Render row-based list with per-item select and always-visible delete
            for it in summaries:
                row_cols = st.sidebar.columns([6, 1])
                label = f'#{it["id"]}-{it["text_preview"][:30]}'
                with row_cols[0]:
                    if st.button(label, key=f"sel_{it['id']}", use_container_width=True):
                        detail = api_get_history_detail(backend_url, st.session_state["username"], int(it["id"]))
                        if detail:
                            st.session_state["essay_text"] = detail.get("text", "")
                            st.session_state["loaded_result"] = {
                                "rating": detail.get("rating"),
                                "probabilities": detail.get("probabilities", []),
                                "concept_predictions": detail.get("concept_predictions"),
                            }
                            # Restore model selection if available
                            try:
                                hist_model = detail.get("model_name")
                                if hist_model and hist_model in AVAILABLE_MODELS:
                                    st.session_state["model_select"] = hist_model
                            except Exception:
                                pass
                            st.rerun()
                with row_cols[1]:
                    if st.button("x", key=f"del_{it['id']}"):
                        if api_delete_history(backend_url, st.session_state["username"], int(it["id"])):
                            st.session_state["history_summaries"] = api_list_history(backend_url, st.session_state["username"])
                            st.session_state.pop("loaded_result", None)
                            st.rerun()

    # Initialize text state
    if "essay_text" not in st.session_state:
        st.session_state["essay_text"] = (
            "Q: What is a pointer in C++?\n"
            "A: A pointer is a variable that stores the memory address of another variable. "
            "It allows indirect access to data and enables dynamic memory management using new/delete."
        )

    # Single-column vertical layout: Enter Essay -> Concept Analysis -> Final Grade

    with st.expander("Enter Essay", expanded=True):
        # Full-width text input
        text_input = st.text_area(
            "Enter Essay",
            value=st.session_state.get("essay_text", ""),
            height=320,
            help="Enter essay text to grade",
            placeholder="Paste or type the essay text here...",
            label_visibility="collapsed",
        )
        st.session_state["essay_text"] = text_input
        # Controls row
        ctrl_left, ctrl_right = st.columns([1, 1])
        # Ensure model select state has a default
        if "model_select" not in st.session_state and AVAILABLE_MODELS:
            st.session_state["model_select"] = AVAILABLE_MODELS[0]
        with ctrl_left:
            model_name = st.selectbox(
                "Model",
                AVAILABLE_MODELS,
                help="Select the model to use for grading",
                label_visibility="collapsed",
                key="model_select",
            )
        with ctrl_right:
            grade_clicked = st.button("Grade", use_container_width=True)
    mode = "joint"

    # Run grading if requested
    result: Optional[Dict[str, Any]] = None
    num_classes: Optional[int] = None
    max_probability: Optional[float] = None
    if grade_clicked and text_input.strip():
        with st.spinner("Analyzing text..."):
            result = predict_single_text(backend_url, text_input, model_name, mode)
        if result:
            # Save to history
            payload = {
                "username": st.session_state["username"],
                "text": text_input,
                "model_name": model_name,
                "mode": mode,
                "prediction": result.get("prediction"),
                "rating": result.get("rating"),
                "probabilities": result.get("probabilities", []),
                "concept_predictions": result.get("concept_predictions"),
            }
            new_id = api_save_grade(backend_url, payload)
            if new_id is not None:
                st.session_state["history_summaries"] = api_list_history(backend_url, st.session_state["username"])
                # Force a rerun so the sidebar history reflects the newly saved record immediately
                # Persist latest result so it stays visible after rerun
                st.session_state["loaded_result"] = {
                    "rating": result.get("rating"),
                    "probabilities": result.get("probabilities", []),
                    "concept_predictions": result.get("concept_predictions"),
                }
                st.rerun()
            num_classes = len(result.get("probabilities", [])) or None
            max_probability = max(result.get("probabilities", [0.0])) if result.get("probabilities") else None

    # If a history record was loaded from sidebar, prepare its values for display
    display_result = result
    if display_result is None and st.session_state.get("loaded_result"):
        display_result = st.session_state.get("loaded_result")
        if display_result:
            num_classes = len(display_result.get("probabilities", [])) or None
            max_probability = max(display_result.get("probabilities", [0.0])) if display_result.get("probabilities") else None

    # Card 2: Concept Analysis (always second)
    if display_result and display_result.get("concept_predictions"):
        st.markdown("### Concept Analysis")
        with st.expander("", expanded=True):
            display_concept_cards(display_result["concept_predictions"], show_header=False)
    # Before grading: do not render Concept Analysis card

    # Card 3: Final Grade (always third; only after grading)
    if display_result and (num_classes is not None) and (max_probability is not None):
        st.markdown("### Final Grade")
        with st.expander("", expanded=True):
            display_rating_compact(display_result["rating"], num_classes, max_probability)

    # Final Grade no longer rendered in the left column


if __name__ == "__main__":
    main()
