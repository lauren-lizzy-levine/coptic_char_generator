import datetime
import logging
import regex as re
import sys

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


def read_datafile(file_name):
    with open(file_name, "r") as f:
        file_text = f.read()
        tt_file = file_text.strip().split("\n")
        logger.info(f"Total lines: {len(tt_file)}")

        sentences = []
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
                        sentences.append(temp_sentence)
                    temp_sentence = temp_orig_group_content
                    new_sentence_detected = False
                else:
                    temp_sentence += temp_orig_group_content
        sentences.append(temp_sentence)
    return sentences


if __name__ == "__main__":
    logger.info(
        f"\nstart coptic data processing -- {datetime.datetime.now()}"
    )

    # TODO point to the external file path
    # read through all the subdirectories

    sentences = read_datafile("./data/AP.001.n135.mother.tt")
    print(sentences)
    logger.info(f"File read: {len(sentences)} sentences")
    # sentences = read_datafile("./data/01_Genesis_03.tt")
    # # print(sentences)
    # logger.info(f"File read: {len(sentences)} sentences")

    # instead of saving in memory, save it into a csv

