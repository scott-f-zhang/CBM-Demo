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
import re
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

def rerun():
    """Helper to handle Streamlit rerun across versions."""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# Available modes
AVAILABLE_MODES = ["joint"]

def get_available_models_from_filesystem() -> list:
    """Dynamically get available models from saved_models/essay directory."""
    try:
        # Use relative path from this file (frontend/app.py) -> project_root -> saved_models/essay
        models_dir = str(Path(__file__).resolve().parent.parent / "saved_models" / "essay")
    except Exception:
        models_dir = "saved_models/essay"

    if not os.path.exists(models_dir):
        return ["roberta-base", "bert-base-uncased"]  # fallback

    # Get all directories in the models folder
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

    # Filter out hidden directories and sort
    model_dirs = [d for d in model_dirs if not d.startswith('.')]
    model_dirs.sort()

    return model_dirs if model_dirs else ["roberta-base", "bert-base-uncased"]  # fallback

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


def api_register(base_url: str, email: str, username: str, password: str) -> bool:
    try:
        resp = requests.post(f"{base_url}/register", json={"email": email, "username": username, "password": password}, timeout=10)
        if resp.status_code == 200:
            return True
        if resp.status_code == 409:
            st.warning("Email already exists.")
            return False
        st.error(f"Register failed: {resp.text}")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return False


def api_login(base_url: str, email: str, password: str) -> bool:
    try:
        resp = requests.post(f"{base_url}/login", json={"email": email, "password": password}, timeout=10)
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


def api_list_history(base_url: str, username: str, limit: int = 20, pinned: bool = False) -> list:
    try:
        resp = requests.get(f"{base_url}/grade_history/list", params={"username": username, "limit": limit, "pinned": pinned}, timeout=10)
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


def predict_with_edited_concepts(base_url: str, text: Optional[str], model_name: str, edited_concepts: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """Send prediction request with edited concept scores to backend."""
    try:
        payload = {
            "text": text,
            "model_name": model_name,
            "mode": "joint",
            "edited_concepts": edited_concepts
        }
        response = requests.post(f"{base_url}/predict-with-concepts", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction with edited concepts failed: {response.text}")
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


def display_concept_cards(concept_predictions: list, show_header: bool = True, editable: bool = False,
                          backend_url: str = "", model_name: str = "", original_text: str = ""):
    """Display concept cards with 2 cards per row, optionally with editing capability."""
    if not concept_predictions:
        return

    if show_header:
        st.markdown('<div style="text-align:center;"><h2>Concept Analysis</h2></div>', unsafe_allow_html=True)

    # Initialize session state for edited concepts if not exists
    if editable and 'edited_concepts' not in st.session_state:
        st.session_state['edited_concepts'] = {}

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

                    # Determine if this concept has been edited
                    edited_concepts_map = st.session_state.get('edited_concepts', {})
                    is_edited = concept_name in edited_concepts_map

                    # Resolve display prediction (original vs edited)
                    display_prediction = prediction
                    if is_edited:
                        edited_val_idx = edited_concepts_map[concept_name]
                        sentiment_labels = list(probs.keys())
                        are_numeric = all(str(k).isdigit() for k in sentiment_labels)

                        if are_numeric:
                            # 0-based index to 1-based label
                            display_prediction = str(edited_val_idx + 1)
                        else:
                            # Text labels mapping
                            if edited_val_idx == 0: display_prediction = "Negative"
                            elif edited_val_idx == 1: display_prediction = "Neutral"
                            elif edited_val_idx == 2: display_prediction = "Positive"

                    # Update icon based on resolved prediction
                    icon = get_icon_for_sentiment(display_prediction)

                    border_color = "#ff6b6b" if is_edited else "transparent"
                    border_width = "2px" if is_edited else "0px"

                    # Create simple card without borders
                    card_html = f"""
                    <div style="
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 5px;
                        background-color: #f8f9fa;
                        border: {border_width} solid {border_color};
                    ">
                        <div style="text-align:center; margin: 0 0 2px 0; color: #333;">
                            <span><strong>{icon} {full_name}</strong></span>
                        </div>
                        <div style="text-align:center; color: #888;">
                            <span><strong>Grading:</strong> {display_prediction}</span>
                            &nbsp; | &nbsp;
                            <span><strong>Top:</strong> {top_sentiment} ({top_prob*100:.1f}%)</span>
                        </div>
                    </div>
                    """

                    st.markdown(card_html, unsafe_allow_html=True)

                    # Add edit functionality if editable
                    if editable:
                        # Get available options from probabilities
                        sentiment_labels = list(probs.keys())
                        are_numeric = all(str(k).isdigit() for k in sentiment_labels)

                        # Create key for this concept's selectbox
                        selectbox_key = f"edit_{concept_name}_{concept_idx}"

                        # Get current value (edited or original)
                        # If edited, convert from 0-based (API) to 1-based (display)
                        edited_value = st.session_state.get('edited_concepts', {}).get(concept_name)
                        if edited_value is not None:
                            # Convert from 0-based API format to 1-based display format
                            current_value = str(edited_value + 1) if are_numeric else prediction
                        else:
                            current_value = prediction

                        # Create selectbox for editing
                        if are_numeric:
                            # For numeric labels (1-5), show as dropdown
                            options = sentiment_labels
                            # Convert current_value to string for comparison
                            current_value_str = str(current_value)
                            try:
                                selected_index = options.index(current_value_str) if current_value_str in options else 0
                            except ValueError:
                                selected_index = 0
                            selected = st.selectbox(
                                f"Edit {concept_name}",
                                options=options,
                                index=selected_index,
                                key=selectbox_key,
                                label_visibility="collapsed"
                            )
                            # Store edited value (convert "1"-"5" to 0-4 for API)
                            # Note: API expects 0-based indices, but we display 1-based labels
                            selected_int = int(selected)
                            # Check if this is different from original (convert original to int for comparison)
                            original_int = int(prediction) if prediction.isdigit() else 0
                            if selected_int != original_int:
                                if 'edited_concepts' not in st.session_state:
                                    st.session_state['edited_concepts'] = {}
                                # Convert "1"-"5" to 0-4 for API
                                st.session_state['edited_concepts'][concept_name] = selected_int - 1
                            elif concept_name in st.session_state.get('edited_concepts', {}):
                                # If user reverts to original, remove from edited dict
                                del st.session_state['edited_concepts'][concept_name]
                        else:
                            # For text labels, show as dropdown
                            options = sentiment_labels
                            selected = st.selectbox(
                                f"Edit {concept_name}",
                                options=options,
                                index=options.index(current_value) if current_value in options else 0,
                                key=selectbox_key,
                                label_visibility="collapsed"
                            )
                            # Store edited value
                            if selected != prediction:
                                if 'edited_concepts' not in st.session_state:
                                    st.session_state['edited_concepts'] = {}
                                # Convert text label to numeric for API
                                if selected == 'Negative':
                                    st.session_state['edited_concepts'][concept_name] = 0
                                elif selected == 'Neutral':
                                    st.session_state['edited_concepts'][concept_name] = 1
                                elif selected == 'Positive':
                                    st.session_state['edited_concepts'][concept_name] = 2
                            elif concept_name in st.session_state.get('edited_concepts', {}):
                                del st.session_state['edited_concepts'][concept_name]

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

    # Add "Repredict" button if editable and there are edits
    if editable and st.session_state.get('edited_concepts'):
        # Add description text above buttons
        st.markdown(
            """
            <div style="margin: 10px 0; padding: 8px; background-color: #f0f2f6; border-radius: 5px;">
                <p style="margin: 0; font-size: 12px; color: #666; line-height: 1.5;">
                    <strong>Regrading:</strong> Re-runs the second stage (C→Y) model using your edited concept scores to generate a new final rating prediction. The original and new predictions will be displayed side-by-side for comparison.<br>
                    <strong>Reset Edits:</strong> Clears all concept score edits and restores the Original Gradings.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col2:
            if st.button("Regrading", use_container_width=True, type="primary"):
                # Call API with edited concepts
                with st.spinner("Repredicting with edited concepts..."):
                    result = predict_with_edited_concepts(
                        backend_url,
                        original_text,
                        model_name,
                        st.session_state['edited_concepts']
                    )
                    if result:
                        # Store result in session state for display
                        st.session_state['reprediction_result'] = result
                        # Set flag to indicate we just performed a regrade
                        st.session_state['just_regraded'] = True
                        rerun()

        # Show reset button
        with col3:
            if st.button("Reset Edits", use_container_width=True):
                st.session_state['edited_concepts'] = {}
                if 'reprediction_result' in st.session_state:
                    del st.session_state['reprediction_result']
                rerun()


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
                st.markdown('<div style="text-align:center;"><h3>Login to Essay CBM</h3></div>', unsafe_allow_html=True)

                def is_valid_email(email: str) -> bool:
                    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
                    return bool(re.match(pattern, email))

                # Form for Login/Register
                # We need tabs or just dynamic rendering based on action
                # Let's keep it simple with side-by-side columns but different state?
                # Actually, standard UI is usually separate tabs.

                # Using session state to toggle between Login and Register
                if "auth_mode" not in st.session_state:
                    st.session_state["auth_mode"] = "login"

                if st.session_state["auth_mode"] == "login":
                    email_login = st.text_input("Email Address", key="login_email")
                    pass_login = st.text_input("Password", type="password", key="login_pass")

                    col_action, col_toggle = st.columns([1, 1])
                    with col_action:
                        if st.button("Login", use_container_width=True):
                            if not email_login or not pass_login:
                                st.error("Please enter both email and password.")
                            elif not is_valid_email(email_login):
                                st.error("Please enter a valid email address.")
                            elif api_login(backend_url, email_login, pass_login):
                                st.session_state["username"] = email_login  # This is now email
                                try:
                                    st.query_params["u"] = email_login
                                except Exception:
                                    try:
                                        st.experimental_set_query_params(u=email_login)
                                    except Exception:
                                        pass
                                rerun()
                    with col_toggle:
                        if st.button("Register", use_container_width=True):
                            st.session_state["auth_mode"] = "register"
                            rerun()

                else:
                    # Register Mode
                    email_reg = st.text_input("Email Address", key="reg_email")
                    username_reg = st.text_input("Username", key="reg_username")
                    pass_reg = st.text_input("Password", type="password", key="reg_pass")

                    col_action, col_toggle = st.columns([1, 1])
                    with col_action:
                        if st.button("Register", use_container_width=True, key="btn_register_action"):
                            if not email_reg or not pass_reg or not username_reg:
                                st.error("All fields are required.")
                            elif not is_valid_email(email_reg):
                                st.error("Please enter a valid email address.")
                            elif api_register(backend_url, email_reg, username_reg, pass_reg):
                                st.success("Registration successful. Please login.")
                                st.session_state["auth_mode"] = "login"
                                rerun()
                    with col_toggle:
                        if st.button("Back to Login", use_container_width=True):
                            st.session_state["auth_mode"] = "login"
                            rerun()
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
        # Assuming we want to show the username if we have it, or the email username part if not.
        # Currently st.session_state["username"] stores the email.
        # We don't have the username stored in session state from login.
        # Ideally, the login response should return the username.
        # For now, we'll stick to the email username part unless we update the login API response.

        display_name = st.session_state["username"]
        if "@" in display_name:
            display_name = display_name.split("@")[0]

        st.sidebar.markdown(
            f'# 👋 Welcome back! {display_name.title()}',
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
            rerun()

        # Pinned Section
        st.sidebar.markdown("### Pinned")
        if "pinned_summaries" not in st.session_state:
            st.session_state["pinned_summaries"] = api_list_history(backend_url, st.session_state["username"], pinned=True)

        pinned_items = st.session_state.get("pinned_summaries", [])
        if pinned_items:
            for it in pinned_items:
                row_cols = st.sidebar.columns([6, 1])
                label = f'📌 #{it["id"]}-{it["text_preview"][:25]}'
                with row_cols[0]:
                    if st.button(label, key=f"sel_pinned_{it['id']}", use_container_width=True):
                        detail = api_get_history_detail(backend_url, st.session_state["username"], int(it["id"]))
                        if detail:
                            st.session_state["essay_text"] = detail.get("text", "")
                            # Restore base result
                            st.session_state["loaded_result"] = {
                                "rating": detail.get("rating"),
                                "probabilities": detail.get("probabilities", []),
                                "concept_predictions": detail.get("concept_predictions"),
                            }

                            # Restore edited concepts and reprediction if they exist
                            edited_concepts = detail.get("edited_concepts")
                            if edited_concepts:
                                st.session_state['edited_concepts'] = edited_concepts
                                # If we have edited concepts, the main rating/probs in the record
                                # correspond to the reprediction. We need to reconstruct the reprediction result object.
                                st.session_state['reprediction_result'] = {
                                    'rating': detail.get("rating"),
                                    'probabilities': detail.get("probabilities", []),
                                    'original_prediction': detail.get("original_prediction"),
                                    'original_rating': detail.get("original_rating"),
                                    'edited_concepts': edited_concepts
                                }
                                # Ensure save button doesn't show up for loaded pinned records
                                st.session_state['just_regraded'] = False
                                # Mark as pinned loaded to disable editing
                                st.session_state['is_pinned_loaded'] = True
                            else:
                                # If no edits, clear any existing edits
                                if 'edited_concepts' in st.session_state:
                                    del st.session_state['edited_concepts']
                                if 'reprediction_result' in st.session_state:
                                    del st.session_state['reprediction_result']
                                st.session_state['is_pinned_loaded'] = True

                            # Restore model selection if available
                            try:
                                hist_model = detail.get("model_name")
                                if hist_model and hist_model in AVAILABLE_MODELS:
                                    st.session_state["model_select"] = hist_model
                            except Exception:
                                pass
                            rerun()
                with row_cols[1]:
                    if st.button("x", key=f"del_pinned_{it['id']}"):
                        if api_delete_history(backend_url, st.session_state["username"], int(it["id"])):
                            st.session_state["pinned_summaries"] = api_list_history(backend_url, st.session_state["username"], pinned=True)
                            st.session_state.pop("loaded_result", None)
                            rerun()

        st.sidebar.markdown("### History")
        if "history_summaries" not in st.session_state:
            st.session_state["history_summaries"] = api_list_history(backend_url, st.session_state["username"], pinned=False)
        # Refresh control
        # ctrl_col1, ctrl_col2 = st.sidebar.columns(2)
        # with ctrl_col1:
        #     if st.sidebar.button("Refresh history"):
        #         st.session_state["history_summaries"] = api_list_history(backend_url, st.session_state["username"], pinned=False)
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

                            # Clear any pinned/edit specific state when loading a regular history item
                            if 'edited_concepts' in st.session_state:
                                del st.session_state['edited_concepts']
                            if 'reprediction_result' in st.session_state:
                                del st.session_state['reprediction_result']
                            if 'is_pinned_loaded' in st.session_state:
                                del st.session_state['is_pinned_loaded']

                            # Restore model selection if available
                            try:
                                hist_model = detail.get("model_name")
                                if hist_model and hist_model in AVAILABLE_MODELS:
                                    st.session_state["model_select"] = hist_model
                            except Exception:
                                pass
                            rerun()
                with row_cols[1]:
                    if st.button("x", key=f"del_{it['id']}"):
                        if api_delete_history(backend_url, st.session_state["username"], int(it["id"])):
                            st.session_state["history_summaries"] = api_list_history(backend_url, st.session_state["username"])
                            st.session_state.pop("loaded_result", None)
                            rerun()

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
                # Ensure we are not in pinned mode for new grades
                if 'is_pinned_loaded' in st.session_state:
                    del st.session_state['is_pinned_loaded']
                rerun()
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
            # Enable editing only if we have text input and model_name, AND it's not a loaded pinned record
            is_pinned = st.session_state.get("is_pinned_loaded", False)
            editable = bool(text_input.strip() and model_name and mode == "joint" and not is_pinned)
            if editable:
                st.markdown(
                    '<p style="font-size: 11px; color: #888; margin-bottom: 10px; font-style: italic;">💡 <strong>Tip:</strong> You can edit concept scores using the dropdown menus below. Edited concepts will be highlighted with a red border. After editing, click "Regrading" to see how the changes affect the final rating.</p>',
                    unsafe_allow_html=True
                )
            elif is_pinned:
                st.markdown(
                    '<p style="font-size: 11px; color: #888; margin-bottom: 10px; font-style: italic;">🔒 <strong>Note:</strong> This is a pinned record and cannot be edited further. To make changes, please create a new grading from the original text.</p>',
                    unsafe_allow_html=True
                )

            # Move the concept cards display here
            display_concept_cards(
                display_result["concept_predictions"],
                show_header=False,
                editable=editable,
                backend_url=backend_url,
                model_name=model_name,
                original_text=text_input if editable else ""
            )

            # Also check for reprediction result and show comparison logic HERE if needed,
            # but typically reprediction UI is separate.

    # Check for reprediction result AFTER concept analysis to ensure button visibility isn't blocked by rerun logic inside components
    reprediction_result = st.session_state.get('reprediction_result')
    if reprediction_result:
        # Show comparison between original and reprediction
        st.markdown("### Regrading Result")
        with st.expander("", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Grading**")
                if reprediction_result.get('original_rating'):
                    display_rating_compact(
                        reprediction_result['original_rating'],
                        num_classes or len(reprediction_result.get('probabilities', [])),
                        max(reprediction_result.get('probabilities', [0.0])) if reprediction_result.get('probabilities') else 0.0
                    )
            with col2:
                st.markdown("**Regrading Result**")
                new_rating = reprediction_result.get('rating')
                new_probs = reprediction_result.get('probabilities', [])
                new_max_prob = max(new_probs) if new_probs else 0.0
                display_rating_compact(
                    new_rating,
                    len(new_probs) or num_classes or 6,
                    new_max_prob
                )

        # Save button for reprediction result - ensure this is outside any columns that might be rebuilt
        # Only show save button if edits are present (i.e. session has edited_concepts)
        # This prevents saving already-pinned records repeatedly unless further edited
        # Also, check if this specific state is already pinned to avoid showing save for a just-loaded pin

        # Logic: Show save if:
        # 1. We have edited concepts (user made changes)
        # 2. We have a reprediction result (Regrading was clicked)
        # 3. Ideally, we don't want to show it if we just loaded a pin.
        #    When we load a pin, we restore 'edited_concepts'.
        #    So we need a way to distinguish "loaded from pin" vs "actively editing".
        #    A simple heuristic is: if we just loaded a pin, maybe we shouldn't show Save immediately.
        #    But if user changes something else, we should.
        #    However, the user requirement is "only when I changed concept score and clicked regrading".
        #    When we load a pin, we haven't just clicked regrading in *this* session, but we restored state.

        # Let's use a session state flag that is set when "Regrading" is actually clicked.
        if st.session_state.get("just_regraded", False):
            if st.button("Save", key="save_reprediction", use_container_width=True):
                # Prepare payload
                base_result = st.session_state.get("loaded_result", {})

                payload = {
                    "username": st.session_state["username"],
                    "text": st.session_state.get("essay_text", ""),
                    "model_name": st.session_state.get("model_select", AVAILABLE_MODELS[0]),
                    "mode": "joint",
                    "prediction": reprediction_result.get("prediction"),
                    "rating": reprediction_result.get("rating"),
                    "probabilities": reprediction_result.get("probabilities", []),
                    "concept_predictions": base_result.get("concept_predictions"),
                    "edited_concepts": st.session_state.get("edited_concepts"),
                    "original_prediction": reprediction_result.get("original_prediction"),
                    "original_rating": reprediction_result.get("original_rating"),
                    "pinned": True
                }

                if api_save_grade(backend_url, payload):
                    st.success("Record pinned successfully!")
                    st.session_state["pinned_summaries"] = api_list_history(backend_url, st.session_state["username"], pinned=True)
                    # Reset the flag so button hides after saving (optional, but good UX)
                    st.session_state["just_regraded"] = False
                    rerun()

        # Update display_result to show reprediction result for final grade card
        display_result = {
            'rating': reprediction_result.get('rating'),
            'probabilities': reprediction_result.get('probabilities', []),
            'concept_predictions': display_result.get('concept_predictions') if display_result else None
        }
        num_classes = len(reprediction_result.get('probabilities', [])) or num_classes
        max_probability = max(reprediction_result.get('probabilities', [0.0])) if reprediction_result.get('probabilities') else max_probability

    # Card 3: Final Grade (always third; only after grading)
    if display_result and (num_classes is not None) and (max_probability is not None):
        st.markdown("### Final Grade")
        with st.expander("", expanded=True):
            display_rating_compact(display_result["rating"], num_classes, max_probability)

    # Final Grade no longer rendered in the left column


if __name__ == "__main__":
    main()
