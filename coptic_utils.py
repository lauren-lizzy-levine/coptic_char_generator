import logging
import os
import sys
import unicodedata
import re
import string
import json
import nltk
from nltk.util import ngrams
from collections import Counter

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
#share = True  # share embedding table with output layer

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

# hidden_size = 100
# hidden_size = 150
# hidden_size = 200
hidden_size = 300
# hidden_size = 400
# hidden_size = 500
# hidden_size = 1000

# rnn_nLayers = 2
# rnn_nLayers = 3
rnn_nLayers = 4

dropout = 0.0
# dropout = 0.1

masking_proportion = 0.15

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
# batch_size_multiplier = 1
batch_size_multiplier = 1.4
# batch_size_multiplier = 1.6
# batch_size_multiplier = 2

# nEpochs = 1
# nEpochs = 2
# nEpochs = 3
# nEpochs = 4
# nEpochs = 10
# nEpochs = 20
# nEpochs = 30
nEpochs = 50

L2_lambda = 0.0
# L2_lambda = 0.001

model_path = "models/"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"torch version & device: {torch.version.__version__, device}")


class DataItem:
    def __init__(self, text=None, indexes=None, mask=None, labels=None):
        self.text = text  # original text
        self.indexes = indexes  # tensor of indexes of characters or tokens
        self.mask = (
            mask  # list of indexes same size as index, true when character is masked
        )
        self.labels = labels  # list of indexes for attention mask


def get_home_path():
    return os.path.expanduser("~")


UNICODE_MARK_NONSPACING = "Mn"
COMBINING_DOT = "COMBINING DOT BELOW"
MN_KEEP_LIST = [COMBINING_DOT]


def filter_diacritics(string):
    new_string = ""
    for character in string:
        if (
            unicodedata.category(character) != UNICODE_MARK_NONSPACING
            or unicodedata.name(character) in MN_KEEP_LIST
        ):
            new_string = new_string + character
    return new_string.lower()


def count_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        if p.dim() > 1:
            logging.debug(f"{p.numel():,}\t{name}")
            total += p.numel()

    logging.info(f"total parameter count = {total:,}")


def read_lacuna_test_files(file_name, data_list):
    with open(f"./data/{file_name}", "r") as f:
        file_text = f.read()
        sentences = file_text.strip().split("\n")

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            data_list.append(sentence)

    return data_list


def read_datafile(file_name, data_list, num_sentences=10):
    with open(file_name, "r") as f:
        file_text = f.read()
        sentences = file_text.strip().split("\n")

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) == 0:
                continue
            data_list.append(DataItem(text=sentence))

            # if len(data_list) > num_sentences:
            #     break

    return data_list


def filter_brackets(input_string):
    input_string = re.sub(r"\|", "", input_string)
    input_string = re.sub(r"\{", "", input_string)
    input_string = re.sub(r"\}", "", input_string)
    input_string = re.sub(r"\(.*\)", "", input_string)
    return input_string


def skip_sentence(input_string):
    skip = False
    if has_more_than_one_latin_character(input_string):
        skip = True
    elif "[----]" in input_string:
        skip = True
    elif len(input_string) < 5:
        skip = True
    return skip


def has_more_than_one_latin_character(input_string):
    latin_count = sum(1 for char in input_string if char in string.ascii_letters)
    return latin_count > 1


def mask_input(model, file_name, mask_type, masking_strategy):
    logger.info(f"Mask type: {mask_type} - {masking_strategy}")

    data = []
    file_path = f"./data/" + file_name
    data = read_datafile(file_path, data)
    logger.info(f"File {file_name} read in with {len(data)} lines")

    data_for_model = []
    mask = False

    if masking_strategy == "once":
        logger.info(f"Masking strategy is {masking_strategy}, masking sentences...")
        for data_item in data:
            masked_data_item, _ = model.mask_and_label_characters(
                data_item, mask_type=mask_type
            )
            data_for_model.append(masked_data_item)
        logger.info("Masking complete")
    elif masking_strategy == "dynamic":
        data_for_model = data
        mask = True

    return data_for_model, mask


def construct_trigram_lookup():
    # read in training data
    with open(f"./data/train.csv", "r") as f:
        file_text = f.read()
        sentences = file_text.strip().split("\n")

    ngram_counts = Counter()
    for sentence in sentences:
        sentence = sentence.strip()
        ngrams_obj = ngrams(sentence, 3, pad_left=True, left_pad_symbol='<s>')
        ngram_counts.update(ngrams_obj)

    look_up = {}
    for entry in ngram_counts:
        first_second = entry[0] + entry[1]
        third = entry[2]
        if first_second in look_up:
            if third in look_up[first_second]:
                # it never should be...
                look_up[first_second][third] += ngram_counts[entry]
            else:
                look_up[first_second][third] = ngram_counts[entry]
        else:
            look_up[first_second] = {entry[2]: ngram_counts[entry]}

    # write look_up tp file
    with open(f"./data/trigram_lookup.json", 'w') as json_file:
        json.dump(look_up, json_file, indent=4)

    return look_up
