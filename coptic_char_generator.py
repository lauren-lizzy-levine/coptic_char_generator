import time
from random import shuffle
import re

from coptic_RNN import *


class DataItem:
    def __init__(
        self, text=None, indexes=None, masked_indexes=None, mask=None, labels=None
    ):
        self.text = text  # original text
        self.indexes = indexes  # tensor of indexes of characters or tokens
        self.mask = (
            mask  # list of indexes same size as index, true when character is masked
        )
        # TODO - i'm not sure what the masks even do, we don't seem to use this?
        self.labels = labels  # list of indexes for attention mask


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
            # For now (prelim model training), skip sentences with real lacunas
            if "[" in sentence:
                continue
            data_list.append(DataItem(text=sentence))
            # print(sentence)
            if len(data_list) > 10000:
                break


def train_batch(model, optimizer, criterion, data, data_indexes, update=True):
    model.zero_grad()
    total_loss, total_tokens, total_chars = 0, 0, 0

    total_masked = 0

    for i in data_indexes:
        data_item = data[i]

        if data_item.indexes is None:
            data_item.indexes = model.lookup_indexes(data_item.text)

        data_item, mask_count = model.mask_and_label_characters(data_item)
        total_masked += mask_count

        # logger.debug("data item ----")
        # logger.debug(f"original text: {data_item.text}")
        # logger.debug(f"orig indexes: {data_item.indexes}")
        # logger.debug(f"self attn labels: {data_item.labels}")
        # logger.debug(f"mask: {data_item.mask}")

        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        label_tensor = torch.tensor(data_item.labels, dtype=torch.int64).to(device)
        out = model([index_tensor])  # [:-1]
        loss = criterion(out[0], label_tensor.view(-1))  # [1:]

        total_loss += loss.data.item()
        total_tokens += len(out[0])
        total_chars += len(data_item.text) + 1

        if update:
            loss.backward()

    if update:
        optimizer.step()

    return total_loss, total_tokens, total_chars, total_masked


def train_model(model, train_data, dev_data=None, output_name="charLM"):
    data_list = [i for i in range(len(train_data))]

    if dev_data is None:
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
        incremental_batch_size = int(bs + 0.5)
        shuffle(train_list)
        model.train()
        train_loss, train_tokens, train_chars, train_mask_count = 0, 0, 0, 0
        for i in range(0, len(train_list), incremental_batch_size):
            loss, num_tokens, num_characters, total_masked = train_batch(
                model,
                optimizer,
                criterion,
                train_data,
                train_list[i : i + incremental_batch_size],
                update=True,
            )
            train_loss += loss
            train_tokens += num_tokens
            train_chars += num_characters
            train_mask_count += total_masked

            if num_characters > 0:
                logger.debug(
                    f"{epoch:4} {i:6} {num_tokens:5} {num_characters:6} loss {loss / num_tokens:7.3f} {loss / num_characters:7.3f} -- tot tr loss: {train_loss / train_tokens:8.4f} {train_loss / train_chars:8.4f}"
                )
        logger.debug(f"masked total: {train_mask_count}")
        model.eval()

        dev_loss, dev_tokens, dev_chars, dev_masked = train_batch(
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
            f"{epoch} tr loss {msg_trn} -- dev loss {msg_dev} -- incremental_batch_size: {incremental_batch_size:4} time elapsed: {time.time() - start:6.1f}"
        )

        logger.info(f"dev masked: {dev_masked}")

        logging.info(f"orig sentence: ⲙ̅ⲡϥ̅ⲟⲩⲱϣⲉϭⲱ̅ϣⲁⲁⲧⲉⲡⲣⲟⲑⲉⲥⲙⲓⲁⲙ̅ⲡⲉϥⲁϩⲉ·")
        prompt = "ⲙ̅ⲡϥ̅ⲟⲩⲱϣ<mask>ϭⲱ̅ϣⲁⲁⲧⲉⲡⲣⲟ<mask>ⲉⲥⲙⲓⲁⲙ̅ⲡⲉϥⲁϩⲉ·"
        sample = fill_masks(model, prompt, temp=0)

        logging.info(
            f"orig sentence: Ⲁϥⲛⲁⲩⲉϩⲏⲗⲓⲁⲥⲉϥⲡⲏⲧ̅ⲁϥⲁⲛⲁⲗⲁⲃⲃⲁⲛⲉⲙ̅ⲙⲟϥⲁϥⲁⲁϥⲛ̅ⲣⲙ̅ⲙ̅ⲡⲉ·"
        )
        prompt = "Ⲁϥⲛⲁⲩⲉϩⲏⲗⲓⲁⲥⲉϥⲡⲏⲧ̅ⲁϥⲁⲛ<mask><mask>ⲁⲃⲃⲁⲛⲉⲙ̅ⲙⲟϥⲁϥⲁⲁ<mask>ⲛ̅ⲣⲙ̅ⲙ̅ⲡⲉ·"
        sample = fill_masks(model, prompt, temp=0)

        logging.info(
            f"orig sentence: Ⲟⲩⲁⲣⲭⲓⲉⲣⲉⲩⲥⲡⲉⲉⲟⲗϫⲉⲛ̅ⲧⲁϥⲧⲁⲗⲟϥⲉϩⲣⲁⲓ̈ϩⲁⲣⲟⲛⲙ̅ⲙⲓⲛⲙ̅ⲙⲟϥ·"
        )
        prompt = "<mask>ⲩⲁⲣⲭⲓⲉⲣⲉⲩⲥ<mask>ⲉⲉⲟⲗϫⲉⲛ̅ⲧⲁϥⲧⲁⲗⲟϥⲉϩⲣⲁⲓ̈ϩⲁⲣⲟⲛⲙ̅ⲙ<mask>ⲛⲙ̅ⲙⲟϥ·"
        sample = fill_masks(model, prompt, temp=0)

        logging.info(
            f"orig sentence: ⲟⲩϩⲟⲟⲩⲇⲉⲉⲃⲟⲗϩⲛⲟⲩϩⲟⲟⲩⲁⲓⲣⲡⲙⲡϣⲁⲁⲡϫ︤ⲥ︥ⲧⲁϩⲙⲉⲧϣⲁⲧⲉⲕⲙⲛⲧⲉⲓⲱⲧ·"
        )
        prompt = "ⲟⲩϩⲟⲟⲩⲇⲉⲉⲃⲟⲗϩⲛⲟⲩϩⲟⲟⲩⲁⲓⲣⲡⲙⲡϣⲁⲁⲡϫ︤ⲥ︥<mask>ⲁϩⲙⲉⲧϣⲁⲧⲉ<mask><mask>ⲛⲧⲉⲓⲱⲧ·"
        sample = fill_masks(model, prompt, temp=0)

        torch.save(model, f"{model_path}/{output_name}.pth")

    accuracy_evaluation(model, dev_data, dev_list)

    return model


def fill_masks(model, text, temp=0):
    logging.info(f"masked prompt: {text}")
    indexes = model.lookup_indexes(text)
    index_tensor = torch.tensor(indexes, dtype=torch.int64).to(device)
    sample_out = model([index_tensor])
    target = []
    for emb in sample_out[0]:
        scores = emb  # [0,-1]
        if temp <= 0:
            _, best = scores.max(0)
            best = best.data.item()
        else:
            output_dist = nn.functional.softmax(scores.view(-1).div(temp))  # .exp()
            best = torch.multinomial(output_dist, 1)[0]
            best = best.data.item()
        target.append(best)
    text = model.decode(target)
    logging.info(f"generated output: {text}")
    return text


def accuracy_evaluation(model, data, data_indexes):
    # first pass at simple accuracy function
    masked_total = 0
    correct = 0
    for i in data_indexes:
        # get model output
        data_item = data[i]
        if data_item.indexes is None:
            data_item.indexes = model.lookup_indexes(data_item.text)
        data_item = model.mask_and_label_characters(data_item)
        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        out = model([index_tensor])

        # get target indexes
        target = []
        for emb in out[0]:
            scores = emb
            _, best = scores.max(0)
            best = best.data.item()
            target.append(best)

        # compare target to label
        # logger.debug(f"self attn labels: {data_item.labels}")
        # logger.debug(f"target labels: {target}")
        assert len(target) == len(data_item.labels)
        for j in range(len(data_item.labels)):
            if data_item.labels[j] > 0:
                # masked token
                masked_total += 1
                if target[j] == data_item.labels[j]:
                    # prediction is correct
                    correct += 1
    logging.info(
        f"masked total: {masked_total}, correct predictions: {correct}, simple accuracy: {correct/masked_total}"
    )
