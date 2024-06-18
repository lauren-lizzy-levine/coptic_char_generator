import torch


class Controller:
    def __init__(self):
        # load model
        model_path = "../models/"
        model_name = "coptic_smart_once_april_best"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = torch.load(model_path + model_name + ".pth", map_location=device)
        self.model = model
