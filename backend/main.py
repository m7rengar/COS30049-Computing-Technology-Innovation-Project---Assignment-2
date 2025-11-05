from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib
import os
from typing import Literal, Optional, List
import numpy as np
from datetime import datetime
import re
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="Machine Learning API for spam message detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (or use ["http://localhost:3000"] for React dev)
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, etc.
    allow_headers=["*"],   # allow all headers
)

# Global variables for models and preprocessors
models = {}
tfidf_vectorizer = None
feature_scaler = None
model_config = None

MODELS_DIR = "models"

# Pydantic Models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text message to classify")
    model: str = Field(
        default="random_forest",
        description="ML model to use for prediction (random_forest or xgboost if available)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You've won a $1000 gift card. Click here to claim now!",
                "model": "random_forest"
            }
        }

class PredictionResponse(BaseModel):
    success: bool
    model_used: str
    prediction: str
    is_spam: bool
    confidence: float
    raw_prediction: int
    timestamp: str

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: str = Field(default="random_forest")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Win a free iPhone now!",
                    "Hey, are we still meeting tomorrow?"
                ],
                "model": "random_forest"
            }
        }

class BatchPredictionResponse(BaseModel):
    success: bool
    model_used: str
    total_predictions: int
    results: List[dict]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    preprocessing_components: dict
    timestamp: str

# Feature Extraction Functions
def extract_additional_features(text: str) -> List[float]:
    """
    Extract 19 additional features from text to match training pipeline
    These are common spam detection features - adjust if your training used different ones
    """
    features = []
    
    # 1. Message length
    features.append(len(text))
    
    # 2. Number of characters
    features.append(len(text))
    
    # 3. Number of words
    features.append(len(text.split()))
    
    # 4. Number of uppercase letters
    features.append(sum(1 for c in text if c.isupper()))
    
    # 5. Number of digits
    features.append(sum(1 for c in text if c.isdigit()))
    
    # 6. Number of special characters
    features.append(sum(1 for c in text if not c.isalnum() and not c.isspace()))
    
    # 7. Number of URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    features.append(len(re.findall(url_pattern, text)))
    
    # 8. Number of email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    features.append(len(re.findall(email_pattern, text)))
    
    # 9. Number of phone numbers (simple pattern)
    phone_pattern = r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b'
    features.append(len(re.findall(phone_pattern, text)))
    
    # 10. Uppercase ratio
    if len(text) > 0:
        features.append(sum(1 for c in text if c.isupper()) / len(text))
    else:
        features.append(0.0)
    
    # 11. Digit ratio
    if len(text) > 0:
        features.append(sum(1 for c in text if c.isdigit()) / len(text))
    else:
        features.append(0.0)
    
    # 12. Special character ratio
    if len(text) > 0:
        features.append(sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text))
    else:
        features.append(0.0)
    
    # 13. Average word length
    words = text.split()
    if len(words) > 0:
        features.append(sum(len(word) for word in words) / len(words))
    else:
        features.append(0.0)
    
    # 14. Number of exclamation marks
    features.append(text.count('!'))
    
    # 15. Number of question marks
    features.append(text.count('?'))
    
    # 16. Number of currency symbols
    features.append(text.count('$') + text.count('£') + text.count('€'))
    
    # 17. Has URL (binary)
    features.append(1.0 if re.search(url_pattern, text) else 0.0)
    
    # 18. Has email (binary)
    features.append(1.0 if re.search(email_pattern, text) else 0.0)
    
    # 19. Has phone number (binary)
    features.append(1.0 if re.search(phone_pattern, text) else 0.0)
    
    return features

# Helper Functions
def load_all_components():
    """Load all models and preprocessing components"""
    global tfidf_vectorizer, feature_scaler, model_config
    
    print("\n" + "="*50)
    print("LOADING SPAM DETECTION MODELS")
    print("="*50)
    
    # Load TF-IDF Vectorizer
    try:
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        tfidf_vectorizer = joblib.load(vectorizer_path)
        vocab_size = len(tfidf_vectorizer.vocabulary_)
        print(f"✓ Loaded TF-IDF vectorizer (vocabulary size: {vocab_size:,})")
    except Exception as e:
        print(f"✗ CRITICAL: Failed to load TF-IDF vectorizer: {e}")
        raise
    
    # Load Feature Scaler
    try:
        scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
        feature_scaler = joblib.load(scaler_path)
        scaler_type = type(feature_scaler).__name__
        expected_features = feature_scaler.n_features_in_
        print(f"✓ Loaded feature scaler ({scaler_type})")
        print(f"  - Expected total features: {expected_features}")
        print(f"  - TF-IDF features: {vocab_size}")
        print(f"  - Additional features: {expected_features - vocab_size}")
    except Exception as e:
        print(f"✗ CRITICAL: Failed to load feature scaler: {e}")
        raise
    
    # Load Model Config (optional)
    try:
        config_path = os.path.join(MODELS_DIR, "model_config.pkl")
        if os.path.exists(config_path):
            model_config = joblib.load(config_path)
            print(f"✓ Loaded model config")
            # Try to print feature names if available
            if isinstance(model_config, dict):
                if 'feature_names' in model_config:
                    print(f"  - Feature names found: {len(model_config['feature_names'])}")
        else:
            print(f"⚠ Model config not found (non-critical)")
    except Exception as e:
        print(f"⚠ Failed to load model config (non-critical): {e}")
    
    # Load ML Models
    model_files = {
        "random_forest": "random_forest_model.pkl",
        "xgboost": "xgboost_model.pkl"
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        try:
            models[model_name] = joblib.load(model_path)
            model_type = type(models[model_name]).__name__
            print(f"✓ Loaded {model_name} ({model_type})")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            if model_name == "xgboost":
                print(f"  ℹ Install xgboost with: pip install xgboost")
    
    print("="*50)
    print(f"READY: {len(models)} models loaded successfully")
    print("="*50 + "\n")

def preprocess_text(text: str) -> np.ndarray:
    """
    Preprocess text using TF-IDF vectorizer, extract additional features, and scale
    
    Args:
        text: Raw text message
        
    Returns:
        Scaled feature array ready for model prediction (3019 features)
    """
    # 1. Transform text to TF-IDF features (3000 features)
    tfidf_features = tfidf_vectorizer.transform([text])
    tfidf_array = tfidf_features.toarray()
    
    # 2. Extract additional features (19 features)
    additional_features = extract_additional_features(text)
    additional_array = np.array(additional_features).reshape(1, -1)
    
    # 3. Combine TF-IDF and additional features (3000 + 19 = 3019 features)
    combined_features = np.hstack([tfidf_array, additional_array])
    
    # 4. Scale features
    scaled_features = feature_scaler.transform(combined_features)
    
    return scaled_features

def get_prediction_details(model, processed_features):
    """
    Get prediction and confidence from model
    
    Returns:
        tuple: (prediction, confidence)
    """
    prediction = model.predict(processed_features)[0]
    
    # Get confidence/probability
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(processed_features)[0]
        confidence = float(max(proba))
    else:
        # For models without predict_proba, use a default
        confidence = 0.85
    
    return int(prediction), float(confidence)

# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Load all components on startup"""
    try:
        load_all_components()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Failed to load required components")
        print(f"Error: {e}")
        print("Server may not function correctly!\n")

# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint - Basic API information
    """
    return {
        "message": "Spam Detection API",
        "status": "running",
        "version": "1.0.0",
        "available_models": list(models.keys()),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "batch_predict": "/predict-batch"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Comprehensive health check endpoint
    """
    return HealthResponse(
        status="healthy" if len(models) > 0 else "unhealthy",
        models_loaded={
            name: {
                "status": "loaded",
                "type": type(model).__name__
            } for name, model in models.items()
        },
        preprocessing_components={
            "tfidf_vectorizer": {
                "status": "loaded" if tfidf_vectorizer else "missing",
                "vocabulary_size": len(tfidf_vectorizer.vocabulary_) if tfidf_vectorizer else 0
            },
            "feature_scaler": {
                "status": "loaded" if feature_scaler else "missing",
                "type": type(feature_scaler).__name__ if feature_scaler else None,
                "expected_features": feature_scaler.n_features_in_ if feature_scaler else 0
            },
            "model_config": {
                "status": "loaded" if model_config else "missing"
            }
        },
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", tags=["Models"])
async def get_models():
    """
    Get list of available models and their details
    """
    model_details = {}
    for name, model in models.items():
        details = {
            "name": name,
            "type": type(model).__name__,
            "has_predict_proba": hasattr(model, 'predict_proba')
        }
        
        # Try to get additional model info
        if hasattr(model, 'n_estimators'):
            details["n_estimators"] = model.n_estimators
        if hasattr(model, 'max_depth'):
            details["max_depth"] = model.max_depth
            
        model_details[name] = details
    
    return {
        "available_models": list(models.keys()),
        "count": len(models),
        "details": model_details,
        "preprocessing_ready": tfidf_vectorizer is not None and feature_scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make spam prediction for a single text message
    
    - **text**: Message to classify
    - **model**: ML model to use (random_forest or xgboost)
    
    Returns prediction with confidence score
    """
    
    # Validate model exists
    if request.model not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not found. Available: {list(models.keys())}"
        )
    
    # Check preprocessing components
    if tfidf_vectorizer is None or feature_scaler is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preprocessing components not loaded. Check server logs."
        )
    
    try:
        # Preprocess the text
        processed_features = preprocess_text(request.text)
        
        # Get the selected model
        model = models[request.model]
        
        # Make prediction
        raw_prediction, confidence = get_prediction_details(model, processed_features)
        
        # Determine spam/ham (assuming 1 = spam, 0 = ham)
        is_spam = bool(raw_prediction == 1)
        prediction_label = "spam" if is_spam else "ham"
        
        return PredictionResponse(
            success=True,
            model_used=request.model,
            prediction=prediction_label,
            is_spam=is_spam,
            confidence=round(confidence, 4),
            raw_prediction=raw_prediction,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple text messages at once
    
    - **texts**: List of messages to classify (max 100)
    - **model**: ML model to use
    
    Returns predictions for all messages
    """
    
    if request.model not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not found"
        )
    
    if tfidf_vectorizer is None or feature_scaler is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preprocessing components not loaded"
        )
    
    try:
        results = []
        model = models[request.model]
        
        for idx, text in enumerate(request.texts):
            processed_features = preprocess_text(text)
            raw_prediction, confidence = get_prediction_details(model, processed_features)
            
            is_spam = bool(raw_prediction == 1)
            
            results.append({
                "index": idx,
                "text_preview": text[:50] + "..." if len(text) > 50 else text,
                "prediction": "spam" if is_spam else "ham",
                "is_spam": is_spam,
                "confidence": round(confidence, 4),
                "raw_prediction": raw_prediction
            })
        
        return BatchPredictionResponse(
            success=True,
            model_used=request.model,
            total_predictions=len(results),
            results=results,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/stats", tags=["General"])
async def get_stats():
    """
    Get API statistics and model information
    """
    stats = {
        "api_version": "1.0.0",
        "models": {
            "total": len(models),
            "available": list(models.keys())
        },
        "preprocessing": {
            "vectorizer": {
                "loaded": tfidf_vectorizer is not None,
                "vocab_size": len(tfidf_vectorizer.vocabulary_) if tfidf_vectorizer else 0
            },
            "scaler": {
                "loaded": feature_scaler is not None,
                "expected_features": feature_scaler.n_features_in_ if feature_scaler else 0
            }
        },
        "feature_pipeline": {
            "tfidf_features": len(tfidf_vectorizer.vocabulary_) if tfidf_vectorizer else 0,
            "additional_features": 19,
            "total_features": (len(tfidf_vectorizer.vocabulary_) + 19) if tfidf_vectorizer else 0
        }
    }
    
    if model_config:
        stats["config"] = model_config if isinstance(model_config, dict) else str(type(model_config))
    
    return stats

@app.get("/test-features", tags=["Debug"])
async def test_features(text: str = "Test message"):
    """
    Debug endpoint to see extracted features for a given text
    """
    try:
        # TF-IDF
        tfidf_features = tfidf_vectorizer.transform([text])
        tfidf_array = tfidf_features.toarray()[0]
        
        # Additional features
        additional_features = extract_additional_features(text)
        
        return {
            "text": text,
            "tfidf_features_shape": tfidf_array.shape,
            "tfidf_features_count": len(tfidf_array),
            "additional_features": additional_features,
            "additional_features_count": len(additional_features),
            "total_features": len(tfidf_array) + len(additional_features),
            "expected_by_scaler": feature_scaler.n_features_in_ if feature_scaler else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction test failed: {str(e)}"
        )

# Run with: uvicorn main:app --reload --port 8000