import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from controller import Controller

import sys
sys.path.append("..")
from coptic_char_generator import predict, predict_top_k, rank
from coptic_utils import DataItem

controller = Controller()

app = FastAPI()
templates = Jinja2Templates(directory="templates/")


@app.get('/')
def read_root(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})


@app.get("/predict")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('predict.html', context={'request': request, 'result': result})


@app.post("/predict")
def form_post(request: Request, sentence: str = Form(...)):
    model = controller.model
    data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
    result = predict(model, data_item)
    return templates.TemplateResponse('predict.html', context={'request': request, 'result': result})


@app.get("/predict_k")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('predict_k.html', context={'request': request, 'result': result})


@app.post("/predict_k")
def form_post(request: Request, sentence: str = Form(...), k: int = Form(...)):
    model = controller.model
    data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
    result = predict_top_k(model, data_item, k)
    return templates.TemplateResponse('predict_k.html', context={'request': request, 'result': result})


@app.get("/rank")
def form_post(request: Request):
    result = ""
    return templates.TemplateResponse('rank.html', context={'request': request, 'result': result})


@app.post("/rank")
def form_post(request: Request, sentence: str = Form(...), options: str = Form(...)):
    model = controller.model
    options = options.split(" ")
    char_indexes = [ind for ind, ele in enumerate(sentence) if ele == "#"]
    ranking = rank(model, sentence, options, char_indexes)
    return templates.TemplateResponse('rank.html', context={'request': request, 'result': ranking})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)