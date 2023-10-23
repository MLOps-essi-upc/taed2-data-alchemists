
from fastapi import FastAPI

app = FastAPI()
model = FastAPI()

@app.get("/")
async def root():
    return {"message": "This is a digit recognizer model. Please update a folder with image of digits and our model will return the prediction of each"}
