import regex as re

from coptic_utils import *


def read_datafiles(file_list):
    sentences = {}

    for file_name in file_list:
        with open(file_name, "r") as f:
            file_text = f.read()
            tt_file = file_text.strip().split("\n")
            logger.debug(f"Total lines in {file_name}: {len(tt_file)}")

            temp_sentence = ""
            temp_orig_group_content = ""

            orig_group = False
            new_sentence_detected = False

            for line in tt_file:
                line = line.strip()
                if line.startswith("<orig_group orig_group=") or line.startswith("<norm_group orig_group="):
                    if line.startswith("<orig_group orig_group="):
                            orig_group = True
                    orig_group_match = re.search("orig_group=\".*?\"", line)
                    # TODO: strip diacritics from content before adding it
                    temp_orig_group_content = orig_group_match.group(0)[12:-1]

                if "new_sent=\"true\"" in line:
                    new_sentence_detected = True

                if (line.startswith("</norm_group") and not orig_group) or line.startswith("</orig_group"):
                    orig_group = False

                    if new_sentence_detected:
                        if len(temp_sentence) > 0:
                            sentences[len(sentences)] = temp_sentence
                        temp_sentence = temp_orig_group_content
                        new_sentence_detected = False
                    else:
                        temp_sentence += temp_orig_group_content
            sentences[len(sentences)] = temp_sentence
    return sentences
