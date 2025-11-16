from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.model_utils import predict

app = FastAPI()

# ===== CORS Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API ROOT =====
@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API Running"}


# ===== PREDICT ROUTE =====
@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Get prediction list from model_utils
    results = predict(image_bytes)

    return {
        "predictions": results
    }


# ===== Run Server Manually =====
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
