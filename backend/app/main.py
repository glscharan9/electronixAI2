from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Define the expert sentiment model we will use from Hugging Face
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# --- FastAPI App ---
app = FastAPI(
    title="High-Accuracy Sentiment Analysis API",
    version="3.0.0"
)

# Add CORS Middleware
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
# Load the high-accuracy model on startup
print(f"Loading expert sentiment model: {MODEL_NAME}...")
classifier = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)
print("Model loaded successfully.")


# --- API Definitions ---
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field must not be empty.")
    
    try:
        # Get the prediction from the pipeline
        prediction = classifier(request.text)[0]
        
        return PredictResponse(
            label=prediction['label'].lower(), 
            score=round(prediction['score'], 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))