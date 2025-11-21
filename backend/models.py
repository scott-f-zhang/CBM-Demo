"""
Pydantic models for request and response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for single text prediction."""
    text: str = Field(..., description="Text to predict on", min_length=1)
    model_name: str = Field(..., description="Model name to use")
    mode: str = Field(..., description="Model mode (standard or joint)")


class ConceptPrediction(BaseModel):
    """Concept prediction for joint mode."""
    concept_name: str
    prediction: str  # "Negative", "Neutral", "Positive"
    probabilities: Dict[str, float]  # {"Negative": 0.1, "Neutral": 0.2, "Positive": 0.7}


class PredictResponse(BaseModel):
    """Response model for single text prediction."""
    prediction: int = Field(..., description="Predicted class (0-4)")
    rating: int = Field(..., description="Predicted rating (1-5 stars)")
    probabilities: List[float] = Field(..., description="Class probabilities")
    concept_predictions: Optional[List[ConceptPrediction]] = Field(
        None, description="Concept predictions (only for joint mode)"
    )


class EvaluateResponse(BaseModel):
    """Response model for batch evaluation."""
    accuracy: float = Field(..., description="Accuracy score")
    macro_f1: float = Field(..., description="Macro F1 score")
    weighted_f1: float = Field(..., description="Weighted F1 score")
    num_samples: int = Field(..., description="Number of samples evaluated")
    predictions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Detailed predictions (if show_details=True)"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    loaded_models: Dict[str, List[str]] = Field(..., description="Loaded models and modes")


class ModelsResponse(BaseModel):
    """Response model for available models."""
    available_models: List[str] = Field(..., description="Available model names")
    available_modes: List[str] = Field(..., description="Available modes")
    loaded_models: Dict[str, List[str]] = Field(..., description="Currently loaded models")


# =========================
# Auth & History Schemas
# =========================

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class SimpleOK(BaseModel):
    ok: bool
    message: Optional[str] = None


class SaveGradeRequest(BaseModel):
    username: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    mode: str = Field(..., min_length=1)
    prediction: int
    rating: int
    probabilities: List[float]
    concept_predictions: Optional[List[ConceptPrediction]] = None
    edited_concepts: Optional[Dict[str, int]] = None
    original_prediction: Optional[int] = None
    original_rating: Optional[int] = None
    pinned: bool = False


class GradeRecordSummary(BaseModel):
    id: int
    created_at: str
    model_name: str
    mode: str
    rating: int
    text_preview: str
    pinned: bool = False


class GradeRecordDetail(BaseModel):
    id: int
    created_at: str
    model_name: str
    mode: str
    prediction: int
    rating: int
    text: str
    probabilities: List[float]
    concept_predictions: Optional[List[ConceptPrediction]] = None
    edited_concepts: Optional[Dict[str, int]] = None
    original_prediction: Optional[int] = None
    original_rating: Optional[int] = None
    pinned: bool = False


class GradeHistoryListResponse(BaseModel):
    items: List[GradeRecordSummary]


class DeleteGradeRequest(BaseModel):
    username: str = Field(..., min_length=1)
    id: int


class PredictWithConceptsRequest(BaseModel):
    """Request model for prediction with edited concept scores."""
    text: Optional[str] = Field(None, description="Original text (optional, for getting original concept predictions)")
    model_name: str = Field(..., description="Model name to use")
    mode: str = Field(..., description="Model mode (must be 'joint')")
    edited_concepts: Dict[str, int] = Field(..., description="Dictionary of edited concept scores, e.g., {'TC': 4, 'UE': 3}")


class PredictWithConceptsResponse(BaseModel):
    """Response model for prediction with edited concepts."""
    prediction: int = Field(..., description="Predicted class (0-4)")
    rating: int = Field(..., description="Predicted rating (1-5 stars)")
    probabilities: List[float] = Field(..., description="Class probabilities")
    original_prediction: Optional[int] = Field(None, description="Original prediction before editing")
    original_rating: Optional[int] = Field(None, description="Original rating before editing")
    edited_concepts: Dict[str, int] = Field(..., description="The edited concept scores that were used")
