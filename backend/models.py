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
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
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


class GradeRecordSummary(BaseModel):
    id: int
    created_at: str
    model_name: str
    mode: str
    rating: int
    text_preview: str


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


class GradeHistoryListResponse(BaseModel):
    items: List[GradeRecordSummary]


class DeleteGradeRequest(BaseModel):
    username: str = Field(..., min_length=1)
    id: int
