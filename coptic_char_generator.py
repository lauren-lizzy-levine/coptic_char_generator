import time, os, sys, random, datetime
from random import shuffle
import re

# import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn

from coptic_RNN import RNN
from coptic_utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(torch.version.__version__, device)

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
    # proj_size = 150
    proj_size = 200
    # proj_size = 250
    # proj_size = 350

# hidden_size = 100
# hidden_size = 150
# hidden_size = 200
# hidden_size = 300
hidden_size = 400
# hidden_size = 500
# hidden_size = 1000

rnn_nLayers = 2
# rnn_nLayers = 3
# rnn_nLayers = 4

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
# learning_rate = 0.0003
learning_rate = 0.001
# learning_rate = 0.003
# learning_rate = 0.01

# initial batchsize
# batch_size = 1
# batch_size = 2
# batch_size = 5
# batch_size = 10
batch_size = 20

# increase the batchsize every epoch by this factor
# batch_size_multiplier = 1
# batch_size_multiplier = 1.4
# batch_size_multiplier = 1.6
batch_size_multiplier = 2

# nEpochs = 1
# nEpochs = 2
nEpochs = 4
# nEpochs = 10
# nEpochs = 20

# L2_lambda = 0.0
L2_lambda = 0.001

model_path = "models/"


class DataItem:
    def __init__(self, text=None, indexes=None, mask=None, labels=None):
        self.text = text  # original text
        self.indexes = indexes  # list of indexes of characters or tokens
        self.mask = (
            mask  # list of indexes same size as index, true when character is masked
        )
        self.labels = labels  # list of indexes attention mask


def count_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        if p.dim() > 1:
            logging.debug(f"{p.numel():,}\t{name}")
            total += p.numel()

    logging.info(f"total parameter count = {total:,}")


def read_datafile(file_name, data_list):
    with open(file_name, "r") as f:
        file_text = f.read()
        sentences = file_text.strip().split("\n")
        sentences = sentences[1:]  # skip header

        for sentence in sentences:
            sentence = sentence.strip()
            sentence = re.sub(r"\d+,", "", sentence)
            if len(sentence) == 0:
                continue
            data_list.append(DataItem(text=sentence))
            # if len(data_list) > 2:
            #     break


def train_batch(model, optimizer, criterion, data, data_indexes, update=True):
    # TODO update loss to be based on accuracy of predictions
    model.zero_grad()
    total_loss, total_tokens, total_chars = 0, 0, 0

    for i in data_indexes:
        data_item = data[i]
        if data_item.indexes is None:
            data_item.indexes = model.lookup_indexes(data_item.text)

        data_item = model.mask_and_label_characters(data_item)

        # logger.debug("data item ----")
        # logger.debug(f"original text: {data_item.text}")
        # logger.debug(f"masked indexes: {data_item.indexes}")
        # logger.debug(f"self attn labels: {data_item.labels}")
        # logger.debug(f"mask: {data_item.mask}")

        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        label_tensor = torch.tensor(data_item.labels, dtype=torch.int64).to(device)
        out, hidden = model([index_tensor])  # [:-1]
        loss = criterion(out[0], label_tensor)  # [1:]

        total_loss += loss.data.item()
        total_tokens += len(out[0])
        total_chars += len(data_item.text) + 1

        if update:
            loss.backward()

    if update:
        optimizer.step()

    return total_loss, total_tokens, total_chars


def train_model(model, train_data, dev_data=None, output_name="charLM"):
    data_list = [i for i in range(len(train_data))]
    if dev_data == None:
        shuffle(data_list)
        num_dev_items = min(int(0.05 * len(train_data)), 2000)
        dev_list = data_list[:num_dev_items]
        train_list = data_list[num_dev_items:]
        dev_data = train_data
    else:
        train_list = data_list
        dev_list = [i for i in range(len(dev_data))]

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=L2_lambda
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    start = time.time()
    bs = batch_size
    for epoch in range(nEpochs):
        if epoch > 0:
            bs *= batch_size_multiplier
        ibs = int(bs + 0.5)
        shuffle(train_list)
        model.train()
        train_loss, train_tokens, train_chars = 0, 0, 0
        for i in range(0, len(train_list), ibs):
            loss, num_tokens, num_characters = train_batch(
                model,
                optimizer,
                criterion,
                train_data,
                train_list[i : i + ibs],
                update=True,
            )
            train_loss += loss
            train_tokens += num_tokens
            train_chars += num_characters

            if num_characters > 0:
                logger.debug(
                    f"{epoch:4} {i:6} {num_tokens:5} {num_characters:6} loss {loss / num_tokens:7.3f} {loss / num_characters:7.3f} -- tot tr loss: {train_loss / train_tokens:8.4f} {train_loss / train_chars:8.4f}"
                )

        model.eval()

        dev_loss, dev_tokens, dev_chars = train_batch(
            model,
            optimizer,
            criterion,
            dev_data,
            dev_list,
            update=False,
        )

        if epoch == 0:
            logger.info(
                f"train={len(train_list):,} {train_tokens:,} {train_chars:,} {train_chars / train_tokens:0.1f} dev={len(dev_list):,} {dev_tokens:,} {dev_chars:,} {dev_chars / dev_tokens:0.1f} bs={batch_size} lr={learning_rate} {model.specs}"
            )

        logging.info(time.time())
        msg_trn = f"{train_loss / train_tokens:8.4f} {train_loss / train_chars:8.4f}"
        msg_dev = f"{dev_loss / dev_tokens:8.4f} {dev_loss / dev_chars:8.4f}"
        logger.info(
            f"{epoch} tr loss {msg_trn} -- dev loss {msg_dev} -- ibs: {ibs:4} time elapsed: {time.time() - start:6.1f}"
        )

        logging.info(f"orig sentence: ⲙ̅ⲡϥ̅ⲟⲩⲱϣⲉϭⲱ̅ϣⲁⲁⲧⲉⲡⲣⲟⲑⲉⲥⲙⲓⲁⲙ̅ⲡⲉϥⲁϩⲉ·")
        seed = "ⲙ̅ⲡϥ̅ⲟⲩ"
        sample = generate(model, seed=seed, n=200, temp=0)
        logging.info(f" genertated output: {sample}")

        torch.save(model, f"{model_path}/{output_name}.pth")

    return model


def generate(model, seed="The ", n=150, temp=0):
    # TODO update this to find <mask> tokens and fill them in instead of generating text out

    model.eval()

    indexes = model.lookup_indexes(seed)
    # adds <s> ... </s> -- remove </s> with [:-1]

    index_list = indexes
    index_tensor = torch.tensor(indexes, dtype=torch.int64).to(device)
    sample_out, sample_hidden = model([index_tensor])

    for i in range(n):
        scores = sample_out[0, -1]
        if temp <= 0:
            _, best = scores.max(0)
            best = best.data.item()
        else:
            output_dist = nn.functional.softmax(scores.view(-1).div(temp))  # .exp()
            best = torch.multinomial(output_dist, 1)[0]
            best = best.data.item()

        index_list.append(best)
        if best == model.eos_id:
            break

        c_in = torch.tensor([best], dtype=torch.int64).to(device)
        c, h = model([c_in], sample_hidden)

    text = model.decode(index_list[1:-1])  # removes <s> & </
    return text
