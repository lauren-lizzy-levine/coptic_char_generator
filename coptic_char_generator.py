import time
from random import shuffle

from coptic_RNN import *
import coptic_utils as utils


def check_accuracy(target, orig_data_item):
    masked = 0
    correct = 0

    if len(target) != len(orig_data_item.labels):
        logging.info("Model predicted different number of characters - sentence skipped")
        return masked, correct
    for j in range(len(orig_data_item.labels)):
        if orig_data_item.labels[j] > 0:
            # masked token
            masked += 1
            # logger.debug(f"actual labels: {orig_data_item.labels[j]}")
            # logger.debug(f"prediction: {target[j]}")
            if target[j] == orig_data_item.labels[j]:
                # prediction is correct
                correct += 1

    return masked, correct


def train_batch(
    model,
    optimizer,
    criterion,
    data,
    data_indexes,
    mask_type,
    update=True,
    mask=True,
):
    model.zero_grad()
    total_loss, total_tokens, total_chars = 0, 0, 0

    total_masked = 0

    dev_masked = 0
    dev_correct = 0

    for i in data_indexes:
        data_item = data[i]

        if mask:
            data_item, mask_count = model.mask_and_label_characters(
                data_item, mask_type=mask_type
            )
            total_masked += mask_count

        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        label_tensor = torch.tensor(data_item.labels, dtype=torch.int64).to(device)
        out = model([index_tensor])  # [:-1]
        # splice out just predicted indexes to go into loss
        masked_idx = torch.BoolTensor(data_item.mask)

        # loss = criterion(out[0], label_tensor.view(-1))  # [1:] old loss method

        masked_out = out[0, masked_idx]
        masked_label = label_tensor[masked_idx]
        # logging.info(f"masked label: {masked_label}")
        loss = criterion(masked_out, masked_label)

        total_loss += loss.data.item()
        total_tokens += len(out[0])
        total_chars += len(data_item.text) + 1

        if update:
            loss.backward()
        else:
            target = []
            for emb in out[0]:
                scores = emb
                _, best = scores.max(0)
                best = best.data.item()
                target.append(best)

            # compare target to label
            # logger.debug(f"self attn labels: {data_item.labels}")
            # logger.debug(f"target labels: {target}")
            # logger.info("No update")
            dev_masked, dev_correct = check_accuracy(target, data_item)

    if update:
        optimizer.step()

    return total_loss, total_tokens, total_chars, total_masked, dev_masked, dev_correct


def train_model(
    model,
    train_data,
    dev_data=None,
    output_name="coptic_lacuna",
    mask=True,
    mask_type=None,
):
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
            loss, num_tokens, num_characters, total_masked, _, _ = train_batch(
                model,
                optimizer,
                criterion,
                train_data,
                train_list[i : i + incremental_batch_size],
                mask_type,
                update=True,
                mask=mask,
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
        (
            dev_loss,
            dev_tokens,
            dev_chars,
            dev_masked,
            dev_masked,
            dev_correct,
        ) = train_batch(
            model,
            optimizer,
            criterion,
            dev_data,
            dev_list,
            mask_type,
            update=False,
            mask=mask,
        )

        if epoch == 0:
            logger.info(
                f"train={len(train_list):,} {train_tokens:,} {train_chars:,} {train_chars / train_tokens:0.1f} "
                f"dev={len(dev_list):,} {dev_tokens:,} {dev_chars:,} {dev_chars / dev_tokens:0.1f} "
                f"bs={batch_size} lr={learning_rate} {model.specs}"
            )

        logging.info(time.time())
        msg_trn = f"{train_loss / train_tokens:8.4f} {train_loss / train_chars:8.4f}"
        msg_dev = f"{dev_loss / dev_tokens:8.4f} {dev_loss / dev_chars:8.4f}"
        logger.info(
            f"{epoch} tr loss {msg_trn} -- dev loss {msg_dev} -- incremental_batch_size: {incremental_batch_size:4} time elapsed: {time.time() - start:6.1f}"
        )

        logging.info(
            f"dev masked total: {dev_masked}, correct predictions: {dev_correct}, simple accuracy: {round(dev_correct / dev_masked, 3)}"
        )

        sample_masked = 0
        sample_correct = 0

        test_sentence = "ϯⲙⲟⲕⲙⲉⲕⲙⲙⲟⲓⲉⲓⲥϩⲉⲛⲣⲟⲙⲡⲉⲉⲧⲙⲧⲣⲉⲣⲱⲙⲉϭⲛϣⲁϫⲉⲉϫⲱⲕⲁⲧⲁⲗⲁⲁⲩⲛⲥⲙⲟⲧ·"
        test_sentence = utils.filter_diacritics(test_sentence)
        _, masked, correct = fill_masks(model, test_sentence, mask_type, temp=0)
        # sample_masked += masked
        # sample_correct += correct

        # logging.info(f"sample accuracy: {round(sample_correct/sample_masked, 3)}")

        torch.save(model, f"{model_path}/{output_name}.pth")

    accuracy_evaluation(model, dev_data, dev_list)
    # baseline_accuracy(model, dev_data, dev_list)

    return model


def fill_masks(model, text, mask_type, temp=0):
    logging.info(f"prompt: {text}")
    test_data_item = DataItem(text=text)
    data_item, _ = model.mask_and_label_characters(test_data_item, mask_type=mask_type)
    index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
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

    target_text = model.decode(target)

    # input vs masked pairs
    pairs = []
    pairs_index = []
    for i in range((len(data_item.mask))):
        if data_item.mask[i]:
            pairs.append((model.decode(data_item.labels[i]), model.decode(target[i])))
            pairs_index.append((data_item.labels[i], target[i]))
    logging.info(f"orig vs predicted char: {pairs}")
    logging.info(f"orig vs predicted char: {pairs_index}")

    sample_masked, sample_correct = check_accuracy(target, test_data_item)
    return target_text, sample_masked, sample_correct


def accuracy_evaluation(model, data, data_indexes):
    # first pass at simple accuracy function
    masked_total = 0
    correct = 0

    for i in data_indexes:
        # get model output
        data_item = data[i]
        data_item.indexes = model.lookup_indexes(data_item.text)
        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        out = model([index_tensor])

        # get target indexes
        target = []
        for emb in out[0]:
            scores = emb
            _, best = scores.max(0)
            best = best.data.item()
            target.append(best)

        # logger.debug("In accuracy")
        masked, correct_guess = check_accuracy(target, data_item)
        masked_total += masked
        correct += correct_guess

    if masked_total > 0:
        logging.info(
            f"dev masked total: {masked_total}, correct predictions: {correct}, simple accuracy: {round(correct/masked_total, 3)}"
        )
    else:
        logging.info(
            f"dev masked total: {masked_total}, correct predictions: {correct}"
        )


def baseline_accuracy(model, data, data_indexes):
    masked_total = 0
    correct_most_common_char = 0
    correct_random = 0
    # Assuming ⲉ is actually the most common...need to confirm with descriptive stats for data
    target_char_index = model.sentence_piece.piece_to_id("ⲉ")
    # logging.info(target_char_index)

    for i in data_indexes:
        data_item = data[i]
        most_common_char_target = [target_char_index] * len(data_item.labels)
        random_target = [
            random.randint(3, model.num_tokens - 1)
            for i in range(len(data_item.labels))
        ]
        _, correct_guess_correct_most_common = check_accuracy(
            most_common_char_target, data_item
        )
        masked, correct_guess_random = check_accuracy(random_target, data_item)
        masked_total += masked
        correct_most_common_char += correct_guess_correct_most_common
        correct_random += correct_guess_random

    logging.info(
        f"Most Common Char Baseline; dev masked total: {masked_total}, correct predictions: {correct_most_common_char}, baseline accuracy: {round(correct_most_common_char / masked_total, 3)}"
    )
    logging.info(
        f"Random Baseline; dev masked total: {masked_total}, correct predictions: {correct_random}, baseline accuracy: {round(correct_random / masked_total, 3)}"
    )


def predict(model, data_item):
    logging.info(f"input text: {data_item.text}")

    index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
    out = model([index_tensor])

    # get target indexes
    target = []
    for emb in out[0]:
        scores = emb
        _, best = scores.max(0)
        best = best.data.item()
        target.append(best)

    out_indexes = []

    for i in range(len(data_item.indexes)):
        if data_item.indexes[i] == model.mask:
            out_indexes.append(target[i])
        else:
            out_indexes.append(data_item.indexes[i])

    out_string = model.decode(out_indexes)

    logging.info(f"output text: {out_string}")
