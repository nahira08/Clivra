from fastapi import FastAPI

app = FastAPI(
    title="Clivra API",
    description="AI Powered Livestream Highlight Detection",
    version="0.1.0"
)

@app.get("/")
def home():
    return{
        "message": "Welcome to Clivra API!",
        "Version": "0.1.0"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "success",
        "message": "Clivra backend is running!"
    }