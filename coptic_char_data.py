import csv
import regex as re

import coptic_utils as utils
import string


def read_datafiles(file_list):
    (
        train_sentences,
        dev_sentences,
        test_sentences,
        empty_lacuna_sentences,
        reconstructed_lacuna_sentences,
    ) = ([], [], [], [], [])
    lacuna_sentences = set()
    reg_sentences = set()
    sentences = collect_sentences(file_list)
    # separate out sentences with actual lacunas
    for sentence in sentences:
        if detect_lacuna(sentence):
            lacuna_sentences.add(sentence)
        else:
            reg_sentences.add(sentence)
    # partition regular sentences
    reg_sentences_list = list(reg_sentences)
    # calculate lengths for each partition based on ratios (90:5:5)
    reg_sent_len = len(reg_sentences_list)
    train_length = int(reg_sent_len * 0.9)
    dev_test_length = int(reg_sent_len * 0.05)
    # partition the list
    train_sentences = reg_sentences_list[:train_length]
    dev_sentences = reg_sentences_list[train_length : train_length + dev_test_length]
    test_sentences = reg_sentences_list[train_length + dev_test_length :]

    # TODO: separate reconstructed vs empty lacuna sentences
    reconstructed_lacuna_sentences = list(lacuna_sentences)

    return (
        train_sentences,
        dev_sentences,
        test_sentences,
        empty_lacuna_sentences,
        reconstructed_lacuna_sentences,
    )


def detect_lacuna(sentence):
    contains_lacuna = False
    lacuna_markers = ["[", "]", "{", "}", "(", ")", "?", "..", "…"]
    if has_more_than_three_latin_characters(sentence):
        contains_lacuna = True
    for marker in lacuna_markers:
        if marker in sentence:
            contains_lacuna = True
            break
    return contains_lacuna


def collect_sentences(file_list):
    sentences = []

    for file_name in file_list:
        with open(file_name, "r") as f:
            file_text = f.read()
            tt_file = file_text.strip().split("\n")
            # logger.debug(f"Total lines in {file_name}: {len(tt_file)}")

            temp_sentence = ""
            temp_orig_group_content = ""

            orig_group = False
            new_sentence_detected = False

            for line in tt_file:
                line = line.strip()
                if line.startswith("<orig_group orig_group=") or line.startswith(
                    "<norm_group orig_group="
                ):
                    if line.startswith("<orig_group orig_group="):
                        orig_group = True
                    orig_group_match = re.search('orig_group=".*?"', line)
                    # TODO: strip diacritics from content before adding it
                    temp_orig_group_content = orig_group_match.group(0)[12:-1]

                if 'new_sent="true"' in line:
                    new_sentence_detected = True

                if (
                    line.startswith("</norm_group") and not orig_group
                ) or line.startswith("</orig_group"):
                    orig_group = False

                    if new_sentence_detected:
                        if len(temp_sentence) > 0:
                            temp_sentence = re.sub(r"|", "", temp_sentence)
                            filtered_sentence = utils.filter_diacritics(temp_sentence)
                            sentences.append(filtered_sentence)
                        temp_sentence = temp_orig_group_content
                        new_sentence_detected = False
                    else:
                        temp_sentence += temp_orig_group_content

            filtered_sentence = utils.filter_diacritics(temp_sentence)

            sentences.append(filtered_sentence)
    return sentences


def has_more_than_three_latin_characters(input_string):
    latin_count = sum(1 for char in input_string if char in string.ascii_letters)
    return latin_count > 3


def write_to_csv(file_name, sentence_list):
    with open(file_name, "w") as csvfile:
        for sentence in sentence_list:
            sent = sentence + "\n"
            csvfile.write(sent)
