"""
FastAPI application for CBM NLP inference service.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    PredictRequest, PredictResponse, EvaluateResponse, 
    HealthResponse, ModelsResponse, ConceptPrediction,
    RegisterRequest, LoginRequest, SimpleOK,
    SaveGradeRequest, GradeRecordSummary, GradeRecordDetail, GradeHistoryListResponse,
    PredictWithConceptsRequest, PredictWithConceptsResponse
)
from model_manager import model_manager
from inference import predict_single, evaluate_batch, predict_with_edited_concepts
from pathlib import Path
import sqlite3
from db import (
    initialize_db,
    create_demo_user,
    create_user,
    verify_user,
    insert_grade_record,
    list_grade_records,
    get_grade_record,
    delete_grade_record,
)

# =========================
# Manual configuration
# =========================
# Select which dataset's models to use (matches folder under `saved_models/`)
CBM_DATASET = "essay"  # e.g., "essay", "imdb", "cebab"

# Make dataset selection visible to the model loader
os.environ["CBM_DATASET"] = CBM_DATASET

# Project root and model discovery
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / "backend" / "data" / "app.db")

def discover_available_models(dataset: str) -> List[str]:
    base_dir = PROJECT_ROOT / "saved_models" / dataset
    if not base_dir.exists():
        return []
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()])


# Create FastAPI app
app = FastAPI(
    title="CBM NLP Inference API",
    description="API for Concept Bottleneck Model inference on text data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available models and modes
AVAILABLE_MODELS = discover_available_models(CBM_DATASET)
AVAILABLE_MODES = ["standard", "joint"]


@app.on_event("startup")
async def startup_event():
    """Load default models at startup."""
    print("Starting CBM NLP API service...")
    print(f"Using device: {model_manager.device}")
    model_manager.load_default_models()
    # Initialize SQLite and seed demo user
    try:
        print(f"Initializing SQLite at {DB_PATH} ...")
        initialize_db(DB_PATH)
        # create_demo_user(DB_PATH) # Disabled: Do not create demo user automatically
        print("Database ready.")
    except Exception as e:
        print(f"Database initialization error: {e}")
    print("API service ready!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    loaded_models = model_manager.get_loaded_models()
    return HealthResponse(
        status="healthy",
        loaded_models=loaded_models
    )

@app.post("/register", response_model=SimpleOK)
async def register(request: RegisterRequest):
    """Register a new user."""
    try:
        create_user(DB_PATH, request.email, request.username, request.password, ignore_exists=False)
        return SimpleOK(ok=True, message="User created")
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login", response_model=SimpleOK)
async def login(request: LoginRequest):
    """Login by verifying email/password."""
    try:
        if verify_user(DB_PATH, request.email, request.password):
            return SimpleOK(ok=True, message="Login success")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade_history/save")
async def save_grade(request: SaveGradeRequest):
    """Save a grading record for the user."""
    try:
        new_id = insert_grade_record(
            DB_PATH,
            request.username,
            text=request.text,
            model_name=request.model_name,
            mode=request.mode,
            prediction=request.prediction,
            rating=request.rating,
            probabilities=request.probabilities,
            concept_predictions=[cp.dict() for cp in (request.concept_predictions or [])],
            edited_concepts=request.edited_concepts,
            original_prediction=request.original_prediction,
            original_rating=request.original_rating,
            pinned=request.pinned,
        )
        return {"id": new_id}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grade_history/list", response_model=GradeHistoryListResponse)
async def grade_history_list(username: str, limit: int = 20, offset: int = 0, pinned: bool = False):
    """List recent grade summaries for a user."""
    try:
        items = list_grade_records(DB_PATH, username, limit=limit, offset=offset, pinned=pinned)
        # Coerce to Pydantic summaries
        summaries = [GradeRecordSummary(**item) for item in items]
        return GradeHistoryListResponse(items=summaries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grade_history/detail", response_model=GradeRecordDetail)
async def grade_history_detail(username: str, id: int):
    """Get detail of a specific grade record for a user."""
    try:
        data = get_grade_record(DB_PATH, username, id)
        if not data:
            raise HTTPException(status_code=404, detail="Record not found")
        # Pydantic will validate and coerce
        return GradeRecordDetail(**data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade_history/delete", response_model=SimpleOK)
async def grade_history_delete(request: Dict[str, Any]):
    """
    Delete a specific grade record for a user.
    Body: { "username": "...", "id": 123 }
    """
    try:
        username = request.get("username")
        record_id = request.get("id")
        if not isinstance(username, str) or not isinstance(record_id, int):
            raise HTTPException(status_code=400, detail="username (str) and id (int) required")
        ok = delete_grade_record(DB_PATH, username, int(record_id))
        if not ok:
            raise HTTPException(status_code=404, detail="Record not found")
        return SimpleOK(ok=True, message="Deleted")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models and currently loaded models."""
    loaded_models = model_manager.get_loaded_models()
    return ModelsResponse(
        available_models=AVAILABLE_MODELS,
        available_modes=AVAILABLE_MODES,
        loaded_models=loaded_models
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment/rating for a single text."""
    # Validate model name
    if request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_name. Available models: {AVAILABLE_MODELS}"
        )
    
    # Validate mode
    if request.mode not in AVAILABLE_MODES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mode. Available modes: {AVAILABLE_MODES}"
        )
    
    try:
        # Perform prediction
        result = predict_single(request.text, request.model_name, request.mode)
        
        # Format concept predictions if present
        concept_predictions = None
        if result['concept_predictions']:
            concept_predictions = [
                ConceptPrediction(**cp) for cp in result['concept_predictions']
            ]
        
        return PredictResponse(
            prediction=result['prediction'],
            rating=result['rating'],
            probabilities=result['probabilities'],
            concept_predictions=concept_predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    mode: str = Form(...),
    show_details: bool = Form(False)
):
    """Evaluate batch of texts with labels from CSV file."""
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_name. Available models: {AVAILABLE_MODELS}"
        )
    
    # Validate mode
    if mode not in AVAILABLE_MODES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mode. Available modes: {AVAILABLE_MODES}"
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a CSV file"
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Validate CSV structure
        if 'text' not in df.columns or 'label' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain 'text' and 'label' columns"
            )
        
        # Extract texts and labels
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        # Validate labels (support 2, 5, and 6 class models)
        max_label = max(labels) if labels else 0
        if max_label > 5:
            raise HTTPException(
                status_code=400, 
                detail="Labels must be integers between 0 and 5 (for 6-class models)"
            )
        elif max_label > 4:
            # 6-class model (0-5)
            pass
        elif max_label > 1:
            # 5-class model (0-4)
            pass
        else:
            # 2-class model (0-1)
            pass
        
        # Perform evaluation
        result = evaluate_batch(texts, labels, model_name, mode, show_details)
        
        return EvaluateResponse(
            accuracy=result['accuracy'],
            macro_f1=result['macro_f1'],
            weighted_f1=result['weighted_f1'],
            num_samples=result['num_samples'],
            predictions=result['predictions']
        )
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/predict-with-concepts", response_model=PredictWithConceptsResponse)
async def predict_with_concepts(request: PredictWithConceptsRequest):
    """Predict final label using edited concept scores, bypassing stage 1 (X->C)."""
    # Validate model name
    if request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_name. Available models: {AVAILABLE_MODELS}"
        )
    
    # Validate mode (must be joint)
    if request.mode != "joint":
        raise HTTPException(
            status_code=400, 
            detail="Mode must be 'joint' for concept editing. Standard mode does not support concepts."
        )
    
    # Validate edited_concepts is not empty
    if not request.edited_concepts:
        raise HTTPException(
            status_code=400,
            detail="edited_concepts cannot be empty. Please provide at least one edited concept score."
        )
    
    try:
        # Perform prediction with edited concepts
        result = predict_with_edited_concepts(
            text=request.text,
            model_name=request.model_name,
            edited_concepts=request.edited_concepts
        )
        
        return PredictWithConceptsResponse(
            prediction=result['prediction'],
            rating=result['rating'],
            probabilities=result['probabilities'],
            original_prediction=result.get('original_prediction'),
            original_rating=result.get('original_rating'),
            edited_concepts=result['edited_concepts']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction with edited concepts failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CBM NLP Inference API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Single text prediction",
            "predict-with-concepts": "POST /predict-with-concepts - Predict with edited concept scores",
            "evaluate": "POST /evaluate - Batch evaluation with CSV upload",
            "health": "GET /health - Health check",
            "models": "GET /models - Available models",
            "docs": "GET /docs - API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
