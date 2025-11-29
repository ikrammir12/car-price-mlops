from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize FastAPI
app = FastAPI(
    title="Car Price Prediction API",
    description="An API to predict used car prices using XGBoost",
    version="1.0.0"
)

# 2. Load the Trained Model
# We load it ONCE when the server starts, not for every request.
try:
    model = joblib.load('model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# 3. Define the Input Data Structure
# This makes sure the user sends the right data types.
class CarInput(BaseModel):
    year: int
    km_driven: int
    fuel: str         # Diesel, Petrol, etc.
    seller_type: str  # Individual, Dealer
    transmission: str # Manual, Automatic
    owner: str        # First Owner, etc.
    mileage: float    # We might need to handle this if your training used it
    engine: float     # same here
    max_power: float  # same here
    seats: float

    # Example for the Swagger UI documentation
    class Config:
        json_schema_extra = {
            "example": {
                "year": 2018,
                "km_driven": 50000,
                "fuel": "Diesel",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner",
                "mileage": 19.0,
                "engine": 1248,
                "max_power": 80.0,
                "seats": 5
            }
        }

# 4. Create the Prediction Endpoint
@app.post("/predict")
def predict_price(car: CarInput):
    if not model:
        return {"error": "Model not loaded"}
    
    # Convert input data to a Pandas DataFrame
    # The pipeline expects a DataFrame with columns matching training
    input_data = pd.DataFrame([car.dict()])
    
    # Make Prediction
    prediction = model.predict(input_data)
    
    # Return result (convert numpy float to python float)
    return {"predicted_price": float(prediction[0])}

# 5. Root Endpoint (Just to check if it works)
@app.get("/")
def home():
    return {"message": "Car Price API is Running! Go to /docs to test it."}