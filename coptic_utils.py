import logging
import os
import sys
import unicodedata

import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler("log/coptic_data_processing.log")
file_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

share = False
# share = True  # share embedding table with output layer

# embed_size = 20
# embed_size = 50
# embed_size = 100
# embed_size = 150
embed_size = 200
# embed_size = 300

if share:
    proj_size = embed_size
else:
    proj_size = 150
    # proj_size = 200
    # proj_size = 250
    # proj_size = 350

hidden_size = 100
# hidden_size = 150
# hidden_size = 200
# hidden_size = 300
# hidden_size = 400
# hidden_size = 500
# hidden_size = 1000

rnn_nLayers = 2
# rnn_nLayers = 3
# rnn_nLayers = 4

dropout = 0.0
# dropout = 0.1

masking_proportion = 0.05

specs = [
    embed_size,
    hidden_size,
    proj_size,
    rnn_nLayers,
    share,
    dropout,
    masking_proportion,
]

# learning_rate = 0.0001
learning_rate = 0.0003
# learning_rate = 0.001
# learning_rate = 0.003
# learning_rate = 0.01

# initial batch size
batch_size = 1
# batch_size = 2
# batch_size = 5
# batch_size = 10
# batch_size = 20

# increase the batch size every epoch by this factor
batch_size_multiplier = 1
# batch_size_multiplier = 1.4
# batch_size_multiplier = 1.6
# batch_size_multiplier = 2

# nEpochs = 1
# nEpochs = 2
# nEpochs = 4
# nEpochs = 10
# nEpochs = 20
nEpochs = 30

L2_lambda = 0.0
# L2_lambda = 0.001

model_path = "models/"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"torch version & device: {torch.version.__version__, device}")


def get_home_path():
    return os.path.expanduser("~")


UNICODE_MARK_NONSPACING = "Mn"
MN_KEEP_LIST = ["COMBINING DOT BELOW"]


def filter_diacritics(string):
    new_string = ""
    for character in string:
        if (
            unicodedata.category(character) != UNICODE_MARK_NONSPACING
            or unicodedata.name(character) in MN_KEEP_LIST
        ):
            new_string = new_string + character
    return new_string.lower()
