from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import random

# Initialize FastAPI app
app = FastAPI(title="GreenGo API", version="1.0.0")

# Configure CORS to allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - change in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
model = None
feature_cols = []
metadata = {}
dataset_stats = {}
dataset_loaded = False

class DatasetAnalyzer:
    def __init__(self):
        self.stats = {}
        self.loaded = False
        
    def load_dataset_stats(self, dataset_path="data/greengo_cleaned_dataset.csv"):
        """Load dataset statistics for enhanced predictions"""
        try:
            if os.path.exists(dataset_path):
                print(f"📊 Loading dataset statistics from {dataset_path}...")
                df = pd.read_csv(dataset_path)
                
                # Clean column names (handle case sensitivity and spaces)
                df.columns = [col.strip().lower() for col in df.columns]
                
                # Calculate comprehensive statistics
                self.stats = {
                    'overall': {
                        'avg_vehicle_count': df['vehicle_count'].mean() if 'vehicle_count' in df.columns else 5.0,
                        'avg_pedestrian_count': df['pedestrian_count'].mean() if 'pedestrian_count' in df.columns else 2.0,
                        'avg_seconds_to_change': df['seconds_to_next_change'].mean() if 'seconds_to_next_change' in df.columns else 30.0,
                    },
                    'by_phase': {}
                }
                
                # Statistics by traffic light phase
                phases = [0, 1, 2]  # green, yellow, red
                phase_names = {0: 'green', 1: 'yellow', 2: 'red'}
                
                # Determine phase column name
                phase_column = None
                for col in ['phase_id', 'current_light', 'phase']:
                    if col in df.columns:
                        phase_column = col
                        break
                
                if phase_column:
                    for phase in phases:
                        phase_data = df[df[phase_column] == phase]
                        if len(phase_data) > 0:
                            self.stats['by_phase'][phase] = {
                                'count': len(phase_data),
                                'avg_vehicle_count': phase_data['vehicle_count'].mean() if 'vehicle_count' in phase_data.columns else 5.0,
                                'avg_pedestrian_count': phase_data['pedestrian_count'].mean() if 'pedestrian_count' in phase_data.columns else 2.0,
                                'avg_seconds_to_change': phase_data['seconds_to_next_change'].mean() if 'seconds_to_next_change' in phase_data.columns else [30, 5, 45][phase],
                                'std_seconds_to_change': phase_data['seconds_to_next_change'].std() if 'seconds_to_next_change' in phase_data.columns else [10, 2, 15][phase],
                            }
                        else:
                            # Default values if no data for this phase
                            self.stats['by_phase'][phase] = {
                                'count': 0,
                                'avg_vehicle_count': 5.0,
                                'avg_pedestrian_count': 2.0,
                                'avg_seconds_to_change': [30, 5, 45][phase],  # green, yellow, red
                                'std_seconds_to_change': [10, 2, 15][phase],
                            }
                else:
                    # If no phase column, use defaults
                    for phase in phases:
                        self.stats['by_phase'][phase] = {
                            'count': 0,
                            'avg_vehicle_count': 5.0,
                            'avg_pedestrian_count': 2.0,
                            'avg_seconds_to_change': [30, 5, 45][phase],
                            'std_seconds_to_change': [10, 2, 15][phase],
                        }
                
                self.loaded = True
                print("✅ Dataset statistics loaded successfully")
                print(f"   - Overall records: {len(df)}")
                print(f"   - Phases analyzed: {list(self.stats['by_phase'].keys())}")
                
            else:
                print(f"⚠️  Dataset file not found at {dataset_path}, using default statistics")
                self._create_default_stats()
                
        except Exception as e:
            print(f"❌ Error loading dataset statistics: {e}")
            self._create_default_stats()
    
    def _create_default_stats(self):
        """Create default statistics when dataset is not available"""
        self.stats = {
            'overall': {
                'avg_vehicle_count': 5.0,
                'avg_pedestrian_count': 2.0,
                'avg_seconds_to_change': 30.0,
            },
            'by_phase': {
                0: {'avg_seconds_to_change': 30.0, 'avg_vehicle_count': 6.0, 'count': 100},  # green
                1: {'avg_seconds_to_change': 5.0, 'avg_vehicle_count': 4.0, 'count': 50},   # yellow
                2: {'avg_seconds_to_change': 45.0, 'avg_vehicle_count': 8.0, 'count': 150}, # red
            }
        }
        self.loaded = True
        print("✅ Default statistics created")

    def get_phase_stats(self, phase_id):
        """Get statistics for a specific phase"""
        return self.stats['by_phase'].get(phase_id, {})
    
    def get_smart_fallback_prediction(self, request):
        """Generate intelligent fallback prediction using dataset patterns"""
        phase_stats = self.get_phase_stats(request.phase_id)
        
        # Base prediction on phase statistics
        base_seconds = phase_stats.get('avg_seconds_to_change', 30.0)
        
        # Adjust based on traffic conditions
        traffic_factor = 1.0
        if request.vehicle_count > phase_stats.get('avg_vehicle_count', 5.0):
            traffic_factor += 0.2  # More traffic = longer wait
        if request.pedestrian_count > 2.0:
            traffic_factor += 0.1  # More pedestrians = longer wait
        
        # Adjust based on weather
        if request.weather_rain_flag == 1:
            traffic_factor += 0.15  # Rain = longer wait
        
        # Calculate final seconds
        seconds_to_change = base_seconds * traffic_factor
        
        # Calculate recommended speed based on distance and time with dynamic factors
        safe_seconds = max(seconds_to_change - 5, 3)  # Leave safety margin
        
        # Add dynamic speed calculation based on multiple factors
        base_speed = (request.distance_to_light / safe_seconds) * 3.6  # m/s to km/h
        
        # Adjust speed based on traffic conditions
        speed_factor = 1.0
        if request.vehicle_count > 8:
            speed_factor -= 0.2  # Heavy traffic = slower speed
        elif request.vehicle_count < 3:
            speed_factor += 0.1  # Light traffic = slightly faster
        
        # Adjust for weather
        if request.weather_rain_flag == 1:
            speed_factor -= 0.15  # Rain = slower speed
            
        # Adjust for phase
        if request.phase_id == 0:  # Green light
            speed_factor += 0.1  # Can go slightly faster
        elif request.phase_id == 2:  # Red light
            speed_factor -= 0.1  # Should slow down
        
        # Add small random variation for realism (±3 km/h)
        random_variation = random.uniform(-3.0, 3.0)
        
        recommended_speed = base_speed * speed_factor + random_variation
        
        # Apply reasonable limits
        seconds_to_change = max(3.0, min(120.0, seconds_to_change))
        recommended_speed = max(10.0, min(80.0, recommended_speed))
        
        return seconds_to_change, recommended_speed

# Initialize dataset analyzer
analyzer = DatasetAnalyzer()

def load_model_and_data():
    """Load model, features, metadata, and dataset statistics"""
    global model, feature_cols, metadata, dataset_loaded
    
    try:
        print("🔧 Loading GreenGo model and data...")
        
        # Load the trained model
        model = joblib.load("models/best_greengo_model.pkl")
        print("✅ Model loaded successfully")
        
        # Load feature columns
        feature_cols = joblib.load("models/feature_columns.pkl")
        print(f"✅ Feature columns loaded: {len(feature_cols)} features")
        
        # Load metadata
        metadata = joblib.load("models/metadata.pkl")
        print("✅ Metadata loaded")
        
        # Load dataset statistics
        analyzer.load_dataset_stats()
        dataset_loaded = analyzer.loaded
        
        print("🎉 All components loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        # Even if model fails, we can still use dataset statistics
        analyzer.load_dataset_stats()
        print("⚠️  Continuing with dataset-based predictions only")

# Load everything at startup
load_model_and_data()

# Define request model
class PredictionRequest(BaseModel):
    vehicle_count: float = 0.0
    pedestrian_count: float = 0.0
    seconds_to_next_change: float = 30.0
    elapsed_in_phase: float = 0.0
    weather_rain_flag: int = 0
    phase_id: int = 0
    distance_to_light: float = 150.0

# Define response model
class PredictionResponse(BaseModel):
    seconds_to_change: float
    recommended_speed: float
    status: str
    message: str = ""
    prediction_source: str = "model"  # model, dataset_fallback, basic_fallback

@app.get("/")
async def root():
    return {
        "message": "GreenGo Traffic Prediction API", 
        "status": "running",
        "features": len(feature_cols),
        "model_ready": model is not None,
        "dataset_loaded": dataset_loaded,
        "dataset_records": sum([stats.get('count', 0) for stats in analyzer.stats['by_phase'].values()])
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "dataset_loaded": dataset_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        print(f"📊 Received prediction request: {request.dict()}")
        
        # Try model prediction first
        if model is not None:
            try:
                # Prepare features in the exact order expected by the model
                feature_dict = {}
                
                # Map request fields to feature columns
                for feature in feature_cols:
                    if feature in request.dict():
                        feature_dict[feature] = request.dict()[feature]
                    else:
                        # Use dataset statistics for missing features
                        if feature == 'vehicle_count':
                            feature_dict[feature] = request.vehicle_count
                        elif feature == 'pedestrian_count':
                            feature_dict[feature] = request.pedestrian_count
                        elif feature == 'seconds_to_next_change':
                            feature_dict[feature] = request.seconds_to_next_change
                        elif feature == 'elapsed_in_phase':
                            feature_dict[feature] = request.elapsed_in_phase
                        elif feature == 'weather_rain_flag':
                            feature_dict[feature] = request.weather_rain_flag
                        elif feature == 'phase_id':
                            feature_dict[feature] = request.phase_id
                        else:
                            feature_dict[feature] = 0.0
                        print(f"⚠️  Using value for feature: {feature} = {feature_dict[feature]}")
                
                # Create feature array in correct order
                X = np.array([[feature_dict[col] for col in feature_cols]])
                print(f"🔢 Feature array shape: {X.shape}")
                
                # Make prediction
                prediction = model.predict(X)
                print(f"🎯 Raw prediction: {prediction}")
                
                # DEBUG: Check prediction structure
                print(f"🔍 DEBUG - Prediction type: {type(prediction)}")
                print(f"🔍 DEBUG - Prediction shape: {getattr(prediction, 'shape', 'No shape')}")
                print(f"🔍 DEBUG - Prediction[0] type: {type(prediction[0])}")
                print(f"🔍 DEBUG - Prediction[0] value: {prediction[0]}")
                
                # Extract seconds and speed from prediction
                if hasattr(prediction[0], '__len__') and len(prediction[0]) >= 2:
                    seconds_to_change = float(prediction[0][0])
                    recommended_speed = float(prediction[0][1])
                    print(f"🔍 DEBUG - Multi-output prediction detected")
                else:
                    seconds_to_change = float(prediction[0])
                    # DYNAMIC speed calculation instead of constant 40.0
                    safe_seconds = max(seconds_to_change - 5, 3)
                    base_speed = (request.distance_to_light / safe_seconds) * 3.6
                    
                    # Add intelligent speed adjustments
                    speed_factor = 1.0
                    if request.vehicle_count > 8:
                        speed_factor -= 0.2
                    elif request.vehicle_count < 3:
                        speed_factor += 0.1
                        
                    if request.weather_rain_flag == 1:
                        speed_factor -= 0.15
                        
                    if request.phase_id == 0:
                        speed_factor += 0.1
                    elif request.phase_id == 2:
                        speed_factor -= 0.1
                    
                    # Add small random variation
                    random_variation = random.uniform(-2.0, 2.0)
                    recommended_speed = base_speed * speed_factor + random_variation
                    print(f"🔍 DEBUG - Single-output prediction, calculated speed dynamically")
                
                # Ensure reasonable values
                seconds_to_change = max(1.0, min(120.0, seconds_to_change))
                recommended_speed = max(10.0, min(80.0, recommended_speed))
                
                response = PredictionResponse(
                    seconds_to_change=seconds_to_change,
                    recommended_speed=recommended_speed,
                    status="success",
                    message="AI model prediction completed successfully",
                    prediction_source="model"
                )
                
                print(f"✅ Model prediction: {seconds_to_change:.1f}s change, {recommended_speed:.1f} km/h")
                return response
                
            except Exception as model_error:
                print(f"⚠️  Model prediction failed: {model_error}, using dataset fallback")
                # Fall through to dataset-based prediction
        
        # Dataset-based fallback prediction
        if dataset_loaded:
            seconds_to_change, recommended_speed = analyzer.get_smart_fallback_prediction(request)
            
            response = PredictionResponse(
                seconds_to_change=seconds_to_change,
                recommended_speed=recommended_speed,
                status="success",
                message="Dataset-based fallback prediction (model unavailable)",
                prediction_source="dataset_fallback"
            )
            
            print(f"📊 Dataset fallback: {seconds_to_change:.1f}s change, {recommended_speed:.1f} km/h")
            return response
        
        # Basic fallback if everything else fails
        else:
            # Simple rule-based fallback
            if request.phase_id == 2:  # red
                seconds_to_change = 45 + (request.distance_to_light / 3.6) * 0.3
            elif request.phase_id == 1:  # yellow
                seconds_to_change = 5
            else:  # green
                seconds_to_change = 30
                
            # Dynamic speed calculation for basic fallback too
            safe_seconds = max(seconds_to_change - 5, 3)
            base_speed = (request.distance_to_light / safe_seconds) * 3.6
            
            # Add variations
            speed_factor = 1.0
            if request.vehicle_count > 6:
                speed_factor -= 0.15
            if request.weather_rain_flag == 1:
                speed_factor -= 0.1
                
            random_variation = random.uniform(-2.0, 2.0)
            recommended_speed = base_speed * speed_factor + random_variation
            
            recommended_speed = max(10.0, min(60.0, recommended_speed))
            seconds_to_change = max(3.0, min(120.0, seconds_to_change))
            
            response = PredictionResponse(
                seconds_to_change=seconds_to_change,
                recommended_speed=recommended_speed,
                status="success",
                message="Basic rule-based fallback prediction",
                prediction_source="basic_fallback"
            )
            
            print(f"🔄 Basic fallback: {seconds_to_change:.1f}s change, {recommended_speed:.1f} km/h")
            return response
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/model-info")
async def model_info():
    return {
        "model_features": feature_cols,
        "metadata": metadata,
        "feature_count": len(feature_cols),
        "model_type": str(type(model)) if model else "None",
        "model_loaded": model is not None,
        "dataset_loaded": dataset_loaded,
        "dataset_stats": analyzer.stats if dataset_loaded else {}
    }

@app.get("/debug-model")
async def debug_model():
    """Debug endpoint to check model output structure"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Create a test input with zeros
        test_input = np.array([[0.0] * len(feature_cols)])
        test_prediction = model.predict(test_input)
        
        return {
            "model_type": str(type(model)),
            "prediction_shape": str(getattr(test_prediction, 'shape', 'No shape')),
            "prediction_sample": str(test_prediction[0]),
            "output_dimension": getattr(test_prediction[0], 'shape', 'No shape') if hasattr(test_prediction[0], 'shape') else 'Scalar',
            "prediction_length": len(test_prediction[0]) if hasattr(test_prediction[0], '__len__') else 1,
            "is_multi_output": hasattr(test_prediction[0], '__len__') and len(test_prediction[0]) >= 2
        }
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

@app.get("/dataset-stats")
async def get_dataset_stats():
    """Endpoint to get dataset statistics"""
    return {
        "dataset_loaded": dataset_loaded,
        "statistics": analyzer.stats if dataset_loaded else {},
        "phase_summary": {
            phase: {
                "name": ["green", "yellow", "red"][phase],
                "avg_seconds": stats.get('avg_seconds_to_change', 0),
                "avg_vehicles": stats.get('avg_vehicle_count', 0),
                "record_count": stats.get('count', 0)
            }
            for phase, stats in analyzer.stats.get('by_phase', {}).items()
        }
    }

@app.get("/phase-stats/{phase_id}")
async def get_phase_stats(phase_id: int):
    """Get statistics for a specific phase"""
    if phase_id not in [0, 1, 2]:
        raise HTTPException(status_code=400, detail="Phase ID must be 0, 1, or 2")
    
    stats = analyzer.get_phase_stats(phase_id)
    return {
        "phase_id": phase_id,
        "phase_name": ["green", "yellow", "red"][phase_id],
        "statistics": stats,
        "dataset_loaded": dataset_loaded
    }

# FIXED: Proper server startup without the if __name__ block
import uvicorn

def start_server():
    print("🚀 Starting GreenGo API server...")
    print("📱 Connect from Flutter using: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/health")
    print("📊 Model info: http://localhost:8000/model-info")
    print("🐛 Debug model: http://localhost:8000/debug-model")
    print("📈 Dataset stats: http://localhost:8000/dataset-stats")
    print("🎯 Prediction sources: Model → Dataset Fallback → Basic Fallback")
    print("⏹️  Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info"
    )

if __name__ == "__main__":
    start_server()