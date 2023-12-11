import csv
import regex as re

import coptic_utils as utils
import unicodedata


def read_datafiles(file_list):
    recon_lacuna_sentences = set()
    empt_lacuna_sentences = set()
    reg_sentences = set()
    sentences = collect_sentences(file_list)
    # separate out sentences with actual lacunas
    for sentence in sentences:
        lacuna_type = detect_lacuna(sentence)
        if lacuna_type:
            if lacuna_type == "reconstructed":
                recon_lacuna_sentences.add(sentence)
            if lacuna_type == "empty":
                empt_lacuna_sentences.add(sentence)
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
    dev_sentences = reg_sentences_list[train_length:train_length + dev_test_length]
    test_sentences = reg_sentences_list[train_length + dev_test_length:]

    filled_reconstructed_lacuna_sentences, masked_reconstructed_lacuna_sentences = mask_lacunas(list(recon_lacuna_sentences))
    _, empty_lacuna_sentences = mask_lacunas(list(empt_lacuna_sentences), reconstruction=False)

    return train_sentences, dev_sentences, test_sentences, empty_lacuna_sentences, \
        filled_reconstructed_lacuna_sentences, masked_reconstructed_lacuna_sentences


def detect_lacuna(sentence):
    lacuna_type = None
    empty_lacuna_markers = ["?", "..", "…", "[.]", "[--]", "[ ]"]
    other_lacuna_markers = ["[", "]", "{", "}", "(", ")"]
    for marker in empty_lacuna_markers:
        if marker in sentence:
            lacuna_type = "empty"
            return lacuna_type
    for marker in other_lacuna_markers:
        if marker in sentence:
            lacuna_type = "reconstructed"
            return lacuna_type
    for character in sentence:
        if unicodedata.name(character) == "COMBINING DOT BELOW":
            lacuna_type = "reconstructed"
            return lacuna_type
    return lacuna_type


def mask_lacunas(sentences, reconstruction=True):
    if reconstruction:
        filled_sentences = []
        masked_sentences = []
        for sentence in sentences:
            filled_sentence = ""
            for character in sentence:
                if not (unicodedata.name(character) == "COMBINING DOT BELOW"\
                        or character == "[" or character == "]"):
                    filled_sentence += character
            temp_masked_sentence = ""
            for character in sentence:
                if unicodedata.name(character) == "COMBINING DOT BELOW":
                    temp_masked_sentence = temp_masked_sentence[:-1] + "#"
                else:
                    temp_masked_sentence += character
            masked_sentence = ""
            in_brackets = False
            for character in temp_masked_sentence:
                if character == "[":
                    in_brackets = True
                elif character == "]":
                    in_brackets = False
                else:
                    if in_brackets:
                        masked_sentence += "#"
                    else:
                        masked_sentence += character
            if not len(set(masked_sentence)) == 1:  # skip completely masked sentences
                filled_sentences.append(filled_sentence)
                masked_sentences.append(masked_sentence)
    else:
        filled_sentences = None
        masked_sentences = []
        for sentence in sentences:
            masked_sentence = ""
            for character in sentence:
                if not (unicodedata.name(character) == "COMBINING DOT BELOW"\
                        or character == "[" or character == "]"):
                    masked_sentence += character
            dot_pattern = r'\.{2,}'  # matches 2 or more consecutive dots
            masked_sentence = re.sub(dot_pattern, lambda match: "#" * len(match.group(0)), masked_sentence)
            masked_sentence = re.sub("…", "#", masked_sentence)
            masked_sentence = re.sub("\?", "#", masked_sentence)
            masked_sentence = re.sub("--", "##", masked_sentence)

            if not len(set(masked_sentence)) == 1: # skip completely masked sentences
                masked_sentences.append(masked_sentence)

    return filled_sentences, masked_sentences


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
                            temp_sentence = utils.filter_brackets(temp_sentence)
                            filtered_sentence = utils.filter_diacritics(temp_sentence)
                            if not utils.skip_sentence(filtered_sentence):
                                sentences.append(filtered_sentence)
                        temp_sentence = temp_orig_group_content
                        new_sentence_detected = False
                    else:
                        temp_sentence += temp_orig_group_content

            temp_sentence = utils.filter_brackets(temp_sentence)
            filtered_sentence = utils.filter_diacritics(temp_sentence)
            if not utils.skip_sentence(filtered_sentence):
                sentences.append(filtered_sentence)
    return sentences


def write_to_csv(file_name, sentence_list):
    with open(file_name, "w") as csvfile:
        for sentence in sentence_list:
            sent = sentence + "\n"
            csvfile.write(sent)
