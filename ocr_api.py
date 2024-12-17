from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ocr import router


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


app.include_router(router, prefix="/api/v1/sparrow-ocr")


@app.get("/")
async def root():
    return {"message": "OCR API"}
