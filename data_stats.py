from collections import Counter
import re

from coptic_utils import logger

FULL_DATA = "./data/full_data.csv"
RECONSTRUCTED_DATA = "./data/masked_test_reconstructed_lacuna.csv"
EMPTY_LACUNA_DATA = "./data/test_empty_lacuna.csv"

FILES = [FULL_DATA, RECONSTRUCTED_DATA, EMPTY_LACUNA_DATA]


def char_histogram(file_name):
    with open(file_name, "r") as f:
        file_text = f.read()
        file_text = file_text.strip()
    char_counts = Counter(file_text)
    return char_counts


def char_counts(file_path):
    logger.info(f"Gathering character stats for {file_path}")
    with open(file_path, "r") as f:
        file_text = f.read()
        sentences = file_text.strip().split("\n")

    logger.info(f"Number of sentences: {len(sentences)}")

    sentences.sort(key=len)

    total_char = 0
    for sentence in sentences:
        total_char += len(sentence)

    logger.info(f"Number of characters: {total_char}")

    logger.info(f"Shortest sentence: {len(sentences[0])}, {sentences[0]}")
    logger.info(f"Longest sentence: {len(sentences[-1])}, {sentences[-1]}")

    ave_sentence_len = total_char / len(sentences)

    logger.info(f"Average sentence length: {round(ave_sentence_len, 2)}")


def gap_counts(file_path):
    logger.info(f"Gathering gap stats for {file_path}")
    with open(file_path, "r") as f:
        file_text = f.read()
        sentences = file_text.strip().split("\n")

    gap_char_count = 0
    for sentence in sentences:
        gap_char_count += sentence.count("#")

    logger.info(f"Gap characters: {gap_char_count}")

    sentences_with_gaps = []
    for sentence in sentences:
        sentences_with_gaps.append(re.findall(r"#+", sentence))

    sentences_with_gaps.sort(key=len)

    masks_per_sentence = {}
    for item in sentences_with_gaps:
        masks_per_sentence.setdefault(len(item), 0)
        masks_per_sentence[len(item)] += 1

    logger.info(f"Masks per sentence: {masks_per_sentence}")

    length_per_gap = {}
    for gap_list in sentences_with_gaps:
        if len(gap_list) > 0:
            for gap in gap_list:
                length_per_gap.setdefault(len(gap), 0)
                length_per_gap[len(gap)] += 1

    sorted_length_per_gap = dict(sorted(length_per_gap.items()))
    logger.info(f"Length per gap: {sorted_length_per_gap}")

    gap_count = 0
    for key, value in sorted_length_per_gap.items():
        gap_count += value

    if gap_count > 0:
        logger.info(
            f"Total gap characters: {gap_char_count}, total gaps: {gap_count}, "
            f"average length per gap {round(gap_char_count/gap_count, 2)}"
        )


if __name__ == "__main__":
    for file in FILES:
        char_histo = char_histogram(file)
        print(char_histo)

        char_counts(file)
        gap_counts(file)
