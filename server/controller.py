import torch


class Controller:
    def __init__(self):
        # load models
        model_path = "../models/"
        smart_once_model_name = "coptic_smart_once_april_best"
        random_dynamic_model_name = "coptic_random_dynamic_5_13"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        smart_once_model = torch.load(model_path + smart_once_model_name + ".pth", map_location=device)
        random_dynamic_model = torch.load(model_path + random_dynamic_model_name + ".pth", map_location=device)
        self.smart_once_model = smart_once_model
        self.random_dynamic_model = random_dynamic_model

        # To Do
        # Re-stylize return data
