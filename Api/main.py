from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Api.routes.fighters import router

app = FastAPI()

# Enable CORS so the frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)

@app.get("/")
def home():
    return {"message": "Welcome to the UFC Fighter Comparison API!"}
