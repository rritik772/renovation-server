import json
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from model_prediction import predict

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", status_code=404)
def dead_root():
    return {}


@app.get("/plan_questions", status_code=200)
def get_plans_questions():
    file_string = open("asset/plan_questions.json", 'r')
    array = json.loads(file_string.read())

    return array


@app.post('/plan_answers')
async def predict_plan_answers(request: Request):
    # print(await request.form())
    formData = jsonable_encoder(await request.form())
    result = predict(formData)
    print(result)

    return {'result': result}
