import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from controller import Controller

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import json

import sys
sys.path.append("..")
from coptic_char_generator import predict, predict_top_k, rank
from coptic_utils import DataItem

controller = Controller()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates/")


@app.get('/')
def read_root(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})


# Custom exception handler for RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors_str = json.dumps(exc.errors())
    return templates.TemplateResponse('error.html', context={'request': request, 'result': errors_str})


# Custom exception handler for generic exceptions (500 Internal Server Error)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return templates.TemplateResponse('error.html', context={'request': request, 'result': exc})


@app.get("/predict")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('predict.html', context={'request': request, 'result': result})


@app.post("/predict")
def form_post(request: Request, sentence: str = Form(...), model_name: str = Form(...)):
    if model_name == "Smart Once":
        model = controller.smart_once_model
    elif model_name == "Random Dynamic":
        model = controller.random_dynamic_model
    data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
    model_output = predict(model, data_item)
    result = {"input": "Input: " + sentence, "output": "Result: " + model_output}
    return templates.TemplateResponse('predict.html', context={'request': request, 'result': result})


@app.get("/predict_k")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('predict_k.html', context={'request': request, 'result': result})


@app.post("/predict_k")
def form_post(request: Request, sentence: str = Form(...), k: int = Form(...), model_name: str = Form(...)):
    if model_name == "Smart Once":
        model = controller.smart_once_model
    elif model_name == "Random Dynamic":
        model = controller.random_dynamic_model
    data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
    model_output = predict_top_k(model, data_item, k)
    string_output = ""
    for k, output in enumerate(model_output):
        string_output += str(k+1) + ". " + output + " "
    result = {"input": "Input: " + sentence, "output": "Result: " + string_output}
    return templates.TemplateResponse('predict_k.html', context={'request': request, 'result': result})


@app.get("/rank")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('rank.html', context={'request': request, 'result': result})


@app.post("/rank")
def form_post(request: Request, sentence: str = Form(...), options: str = Form(...), model_name: str = Form(...)):
    if model_name == "Smart Once":
        model = controller.smart_once_model
    elif model_name == "Random Dynamic":
        model = controller.random_dynamic_model
    options = options.split("|")
    char_indexes = [ind for ind, ele in enumerate(sentence) if ele == "#"]
    model_output = rank(model, sentence, options, char_indexes)
    string_output = ""
    for k, output in enumerate(model_output):
        string_output += str(k + 1) + ". " + output[0] + " "
    result = {"input": "Input: " + sentence, "output": "Result: " + string_output}
    return templates.TemplateResponse('rank.html', context={'request': request, 'result': result})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)