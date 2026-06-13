from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import joblib
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Crop Recommendation API",
    description="API for recommending crops based on location and soil data",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Setup static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request and response models
class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    location_name: Optional[str] = None
    use_llm: bool = True

class CropRecommendation(BaseModel):
    crop: str
    confidence: float

class RecommendationResponse(BaseModel):
    location: Dict[str, Any]
    soil_characteristics: Dict[str, Any]
    weather_data: Optional[Dict[str, Any]]
    crop_recommendations: List[Dict[str, Any]]
    process_log: List[str]

# Global variables for loaded models
soil_models = None
soil_encoders = None
soil_scaler = None
cat_model = None
scaler_cat = None
mlb = None
weather_api_key = os.getenv('WEATHER_API_KEY')


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


# Load all models at startup
@app.on_event("startup")
async def load_models():
    global soil_models, soil_encoders, soil_scaler, cat_model, scaler_cat, mlb
    
    try:
        # Set the models directory path
        models_dir = "models"
        
        # Load soil prediction models
        soil_models = joblib.load(os.path.join(models_dir, 'soil_prediction_models.pkl'))
        soil_encoders = joblib.load(os.path.join(models_dir, 'soil_level_encoders.pkl'))
        soil_scaler = joblib.load(os.path.join(models_dir, 'soil_feature_scaler.pkl'))
        
        # Load crop recommendation models
        cat_model = joblib.load(os.path.join(models_dir, 'crop_recommender_model.pkl'))
        scaler_cat = joblib.load(os.path.join(models_dir, 'crop_feature_scaler.pkl'))
        mlb = joblib.load(os.path.join(models_dir, 'crop_multilabel_binarizer.pkl'))
        
        # Load metadata for informational purposes
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        print(f"All models loaded successfully. Supporting {len(metadata['crop_classes'])} crops.")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e
    
# Add these functions after the load_models function

# Predict soil characteristics based on location
def predict_soil_characteristics(latitude, longitude):
    """
    Predict soil characteristics (N, P, K, pH levels) based on location
    """
    # Prepare input data
    input_location = np.array([[latitude, longitude]])
    
    # Scale the input
    input_scaled = soil_scaler.transform(input_location)
    
    # Make predictions for each soil characteristic
    predictions = {}
    
    for level in soil_encoders.keys():
        # Predict encoded level
        level_encoded = soil_models[level].predict(input_scaled)[0]
        
        # Decode to get the actual level name
        level_name = soil_encoders[level].inverse_transform([level_encoded])[0]
        
        # Store in predictions dictionary
        predictions[level] = level_name
    
    return predictions

# Get weather data
async def get_weather_data(latitude, longitude):
    """
    Fetch weather data from a weather API for a specific location
    """
    url = f"https://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={latitude},{longitude}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant weather information
            weather_data = {
                "lat": latitude,
                "lon": longitude,
                "temperature": data["current"]["temp_c"],
                "humidity": data["current"]["humidity"],
                "description": data["current"]["condition"]["text"],
                "wind_speed": data["current"]["wind_kph"] / 3.6,  # Convert to m/s
                "rainfall": data["current"]["precip_mm"],
                "timestamp": datetime.now().isoformat()
            }
            return weather_data
        else:
            raise Exception(f"Error fetching weather data: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
        return None

# Convert soil level to distribution
def convert_level_to_distribution(level, is_ph=False):
    """
    Convert categorical soil level to numeric distribution
    """
    if not is_ph:
        # For NPK levels
        if level == 'High':
            return (0.7, 0.2, 0.1)  # (high, medium, low)
        elif level == 'Medium':
            return (0.3, 0.6, 0.1)  # (high, medium, low)
        elif level == 'Low':
            return (0.1, 0.2, 0.7)  # (high, medium, low)
        else:
            return (0.33, 0.33, 0.34)  # balanced if unknown
    else:
        # For pH levels (acidic, neutral, alkaline)
        if level == 'Acidic':
            return (0.7, 0.2, 0.1)  # (acidic, neutral, alkaline)
        elif level == 'Neutral':
            return (0.1, 0.8, 0.1)  # (acidic, neutral, alkaline)
        elif level == 'Alkaline':
            return (0.1, 0.2, 0.7)  # (acidic, neutral, alkaline)
        else:
            return (0.33, 0.33, 0.34)  # balanced if unknown

# Recommend crops based on soil characteristics and location
def recommend_crops(latitude, longitude, 
                   nitrogen_levels, phosphorous_levels, potassium_levels, ph_levels,
                   top_n=5):
    """
    Recommend crops based on location and soil characteristics
    """
    # Prepare input features
    # Convert location to scaled format
    loc_input = np.array([[latitude, longitude]])
    scaled_loc = scaler_cat.transform(loc_input)
    
    # Create feature vector with soil characteristics distributions
    features = []
    
    # Add scaled location
    features.extend(scaled_loc[0])
    
    # Add one-hot encoded distributions for N levels (High, Medium, Low)
    features.extend([
        nitrogen_levels[0],     # N_High
        nitrogen_levels[1],     # N_Medium  
        nitrogen_levels[2]      # N_Low
    ])
    
    # Add one-hot encoded distributions for P levels
    features.extend([
        phosphorous_levels[0],  # P_High
        phosphorous_levels[1],  # P_Medium
        phosphorous_levels[2]   # P_Low
    ])
    
    # Add one-hot encoded distributions for K levels
    features.extend([
        potassium_levels[0],    # K_High
        potassium_levels[1],    # K_Medium
        potassium_levels[2]     # K_Low
    ])
    
    # Add one-hot encoded distributions for pH levels
    features.extend([
        ph_levels[0],           # pH_Acidic
        ph_levels[1],           # pH_Neutral
        ph_levels[2]            # pH_Alkaline
    ])
    
    # Convert to numpy array and reshape for prediction
    X_pred = np.array(features).reshape(1, -1)
    
    # Predict probability for each crop using the categorical model
    crop_probabilities = cat_model.predict_proba(X_pred)
    
    # Get the classes (crops) from the MultiLabelBinarizer
    crop_names = mlb.classes_
    
    # Create list of (crop, probability) pairs
    crop_prob_pairs = []
    
    # For each classifier in the MultiOutputClassifier
    for i, proba_list in enumerate(crop_probabilities):
        # If the crop has a probability > 0 for class 1 (presence of crop)
        if len(proba_list[0]) > 1:  # Make sure we have probabilities for both classes
            # Get probability for class 1 (presence of crop)
            prob = proba_list[0][1]
            crop_name = crop_names[i]
            crop_prob_pairs.append((crop_name, prob))
    
    # Sort by probability in descending order
    crop_prob_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return crop_prob_pairs[:top_n]

# Add these functions after the recommend_crops function

# Import for LLM
try:
    from openai import OpenAI
    ENABLE_LLM = True
    
    # LLM settings
    token = os.getenv("GITHUB_TOKEN")  # Get token from .env
    endpoint = os.getenv("LLM_ENDPOINT", "https://models.inference.ai.azure.com")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    
    if token:
        client = OpenAI(
            base_url=endpoint,
            api_key=token,
        )
    else:
        ENABLE_LLM = False
        print("Warning: LLM token not found in environment variables. LLM validation will be disabled.")
except ImportError:
    ENABLE_LLM = False
    print("Warning: OpenAI package not installed. LLM validation will be disabled.")

async def call_llm_api(prompt):
    """
    Call LLM API with the given prompt
    """
    if not ENABLE_LLM:
        # Return a fallback response in valid JSON format
        return json.dumps({
            "N_level": "Medium", 
            "P_level": "Medium", 
            "K_level": "Low", 
            "pH_level": "Neutral",
            "explanation": "LLM validation is not enabled."
        })
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an agricultural expert who specializes in soil science."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        # Return a fallback response in valid JSON format
        return json.dumps({
            "N_level": "Medium", 
            "P_level": "Medium", 
            "K_level": "Low", 
            "pH_level": "Neutral",
            "explanation": "Using default values due to API error."
        })

async def validate_soil_with_llm(soil_predictions, weather_data, location_name=None):
    """
    Send predicted soil characteristics and weather data to LLM for validation/adjustment
    """
    # Make a copy of the predictions to avoid modifying the original
    soil_predictions = soil_predictions.copy()
    
    # Handle NaN values and set defaults
    for key in ['N_level', 'P_level', 'K_level', 'pH_level']:
        if key not in soil_predictions or pd.isna(soil_predictions[key]):
            if key == 'N_level':
                soil_predictions[key] = 'Medium'
            elif key == 'P_level':
                soil_predictions[key] = 'Medium'  
            elif key == 'K_level':
                soil_predictions[key] = 'Low'
            elif key == 'pH_level':
                soil_predictions[key] = 'Neutral'
                
    # Add default numeric values if they don't exist
    # These are approximate values based on the categorical levels
    if 'N' not in soil_predictions:
        # Add estimated values based on levels
        if soil_predictions['N_level'] == 'High':
            soil_predictions['N'] = 120
        elif soil_predictions['N_level'] == 'Medium':
            soil_predictions['N'] = 80
        else:  # Low
            soil_predictions['N'] = 40
            
    if 'P' not in soil_predictions:
        if soil_predictions['P_level'] == 'High':
            soil_predictions['P'] = 80
        elif soil_predictions['P_level'] == 'Medium':
            soil_predictions['P'] = 40
        else:  # Low
            soil_predictions['P'] = 20
            
    if 'K' not in soil_predictions:
        if soil_predictions['K_level'] == 'High':
            soil_predictions['K'] = 240
        elif soil_predictions['K_level'] == 'Medium':
            soil_predictions['K'] = 170
        else:  # Low
            soil_predictions['K'] = 100
            
    if 'ph' not in soil_predictions:
        if soil_predictions['pH_level'] == 'Acidic':
            soil_predictions['ph'] = 5.5
        elif soil_predictions['pH_level'] == 'Neutral':
            soil_predictions['ph'] = 7.0
        else:  # Alkaline
            soil_predictions['ph'] = 8.5
    
    # Format the prompt for the LLM
    location_text = f"Location: {location_name}" if location_name else f"Coordinates: ({weather_data['lat']}, {weather_data['lon']})"
    
    prompt = f"""
    I need to validate soil characteristics for agricultural planning.
    
    {location_text}
    
    Weather conditions:
    - Temperature: {weather_data['temperature']}°C
    - Humidity: {weather_data['humidity']}%
    - Weather description: {weather_data['description']}
    - Wind speed: {weather_data['wind_speed']} m/s
    - Recent rainfall: {weather_data['rainfall']} mm
    
    Machine learning model predictions for soil characteristics:
    - Nitrogen level: {soil_predictions['N_level']} (N: {soil_predictions['N']} mg/kg)
    - Phosphorous level: {soil_predictions['P_level']} (P: {soil_predictions['P']} mg/kg)
    - Potassium level: {soil_predictions['K_level']} (K: {soil_predictions['K']} mg/kg)
    - pH level: {soil_predictions['pH_level']} (pH: {soil_predictions['ph']})
    
    Based on the location and weather conditions, validate these soil characteristic predictions. 
    If you believe adjustments are needed, explain why and provide the adjusted values.
    Return your response in this JSON format:
    {{
        "N_level": "value",
        "P_level": "value", 
        "K_level": "value",
        "pH_level": "value",
        "N": numeric_value,
        "P": numeric_value,
        "K": numeric_value,
        "ph": numeric_value,
        "explanation": "your reasoning"
    }}
    """
    
    # Call LLM API
    response = await call_llm_api(prompt)
    
    try:
        # The issue might be that the response includes the ```json and ``` markers
        # Let's clean the response before parsing
        cleaned_response = response
        if "```json" in response:
            # Extract just the JSON part
            cleaned_response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            # In case the response is wrapped in triple backticks without specifying json
            cleaned_response = response.split("```")[1].strip()
        
        validated_soil = json.loads(cleaned_response)
        return validated_soil
    except Exception as e:
        # Log the specific error to help with debugging
        print(f"Warning: LLM response could not be parsed: {str(e)}")
        print("Using original predictions")
        return soil_predictions
    
# Add this endpoint after all the helper functions

@app.post("/recommend", response_model=RecommendationResponse)
async def get_crop_recommendations(request: LocationRequest):
    """
    Get crop recommendations based on location
    
    This endpoint takes a location (latitude/longitude) and provides:
    - Soil characteristics based on the location
    - Weather data for the location
    - LLM validation of soil data (if enabled)
    - Crop recommendations based on the soil and location
    """
    results = {
        "location": {
            "latitude": request.latitude,
            "longitude": request.longitude,
            "name": request.location_name
        },
        "soil_characteristics": {},
        "weather_data": {},
        "crop_recommendations": [],
        "process_log": []
    }
    
    # Step 1: Predict soil characteristics from location
    try:
        soil_predictions = predict_soil_characteristics(request.latitude, request.longitude)
        results["soil_characteristics"]["predicted"] = soil_predictions
        results["process_log"].append("Soil characteristics predicted from location model")
    except Exception as e:
        error_msg = f"Error predicting soil characteristics: {str(e)}"
        results["process_log"].append(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Step 2: Get weather data
    try:
        weather_data = await get_weather_data(request.latitude, request.longitude)
        results["weather_data"] = weather_data
        results["process_log"].append("Weather data retrieved successfully")
    except Exception as e:
        results["process_log"].append(f"Weather data retrieval failed: {str(e)}")
        request.use_llm = False  # Can't use LLM without weather data
    
    # Step 3: LLM validation if enabled
    soil_to_use = soil_predictions
    if request.use_llm and ENABLE_LLM and weather_data:
        try:
            validated_soil = await validate_soil_with_llm(soil_predictions, weather_data, request.location_name)
            results["soil_characteristics"]["validated"] = validated_soil
            results["process_log"].append("Soil characteristics validated by LLM")
            
            # Use validated soil for crop recommendations
            soil_to_use = validated_soil
        except Exception as e:
            results["process_log"].append(f"LLM validation failed: {str(e)}")
    
    # Handle missing or NaN values in soil_to_use
    for key in ['N_level', 'P_level', 'K_level', 'pH_level']:
        if key not in soil_to_use or pd.isna(soil_to_use[key]):
            if key == 'N_level':
                soil_to_use[key] = 'Medium'
            elif key == 'P_level':
                soil_to_use[key] = 'Medium'  
            elif key == 'K_level':
                soil_to_use[key] = 'Low'
            elif key == 'pH_level':
                soil_to_use[key] = 'Neutral'
    
    # Step 4: Convert soil levels to distributions for crop recommender
    try:
        n_level_dist = convert_level_to_distribution(soil_to_use['N_level'])
        p_level_dist = convert_level_to_distribution(soil_to_use['P_level'])
        k_level_dist = convert_level_to_distribution(soil_to_use['K_level'])
        ph_level_dist = convert_level_to_distribution(soil_to_use['pH_level'], is_ph=True)
        
        # Step 5: Get crop recommendations
        recommendations = recommend_crops(
            request.latitude, request.longitude,
            nitrogen_levels=n_level_dist,
            phosphorous_levels=p_level_dist,
            potassium_levels=k_level_dist,
            ph_levels=ph_level_dist,
            top_n=10
        )
        
        results["crop_recommendations"] = [
            {"crop": crop, "confidence": float(conf)} 
            for crop, conf in recommendations
        ]
        results["process_log"].append("Crop recommendations generated successfully")
        
    except Exception as e:
        error_msg = f"Error generating crop recommendations: {str(e)}"
        results["process_log"].append(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    return results

# Add these endpoints after the /recommend endpoint

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    model_status = all([soil_models, soil_encoders, soil_scaler, cat_model, scaler_cat, mlb])
    return {
        "status": "healthy", 
        "models_loaded": model_status,
        "llm_enabled": ENABLE_LLM
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the models used by the API"""
    try:
        # Load the metadata file for model information
        with open(os.path.join("models", "model_metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Return detailed model information
        return {
            "version": metadata["version"],
            "created_date": metadata["created_date"],
            "description": metadata["model_description"],
            "pipeline": metadata["pipeline_steps"],
            "supported_crops_count": len(metadata["crop_classes"]),
            "sample_crops": metadata["crop_classes"][:10] + ["..."] if len(metadata["crop_classes"]) > 10 else metadata["crop_classes"],
            "soil_model_accuracy": metadata["soil_model_accuracy"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")
    
# Add this at the end of the file

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)