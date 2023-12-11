import re


with open("./coptic_char_generator/data/full_data.csv", "r") as f:
    file_text = f.read()
    sentences = file_text.strip().split("\n")

sentences.sort(key=len)
len(sentences[0])
5
len(sentences[-1])
1067
total_char = 0
for sentence in sentences:
    total_char += len(sentence)

ave_sentence_len = total_char / len(sentences)
with open("./coptic_char_generator/data/masked_test_reconstructed_lacuna.csv", "r") as f_masked:
    file_text = f_masked.read()
    sentences_masked = file_text.strip().split("\n")

len(sentences_masked)
792
total_char_masked = 0
for sentence in sentences_masked:
    total_char_masked += len(sentence)

underdot_count_masked = 0
for sentence in sentences_masked:
    underdot_count_masked += sentence.count("̣")

with open("./coptic_char_generator/data/test_empty_lacuna.csv", "r") as f_empty:
    file_text = f_empty.read()
    sentences_empty = file_text.strip().split("\n")

sentences_masked.sort(key=len)
sentences_empty.sort(key=len)
len(sentences_masked[0])
3
len(sentences_masked[-1])
459
len(sentences_empty[-1])
362
len(sentences_empty[0])
3

total_char_empty = 0
for sentence in sentences_empty:
    total_char_empty += len(sentence)

gap_masked_count = 0
gap_empty_count = 0
for sentence in sentences_masked:
    gap_masked_count += sentence.count("#")

for sentence in sentences_empty:
    gap_empty_count += sentence.count("#")

sentences_masked_gaps = []
for sentence in sentences_masked:
    sentences_masked_gaps.append(re.findall(r"\#+", sentence))
sentences_masked_gaps.sort(key=len)

masks_per_sentence = {}
for item in sentences_masked_gaps:
    masks_per_sentence.setdefault(len(item), 0)
masks_per_sentence[len(item)] += 1
masks_per_sentence
{0: 8, 1: 501, 2: 154, 3: 49, 4: 26, 5: 16, 6: 10, 7: 10, 8: 6, 9: 4, 10: 1, 12: 1, 13: 2, 14: 1, 15: 1, 19: 1, 20: 1}

masks_per_sentence_empty = {}
for item in sentences_empty:
    masks_per_sentence_empty.setdefault(len(item), 0)
masks_per_sentence_empty[len(item)] += 1

sentences_empty.sort(key=len)
masks_per_sentence_empty = {}
for item in sentences_empty:
    masks_per_sentence_empty.setdefault(len(item), 0)
masks_per_sentence_empty[len(item)] += 1

sentences_empty_gaps = []
for sentence in sentences_empty:
    sentences_empty_gaps.append(re.findall(r"\#+", sentence))

sentences_empty_gaps.sort(key=len)
masks_per_sentence_empty = {}
for item in sentences_empty_gaps:
    masks_per_sentence_empty.setdefault(len(item), 0)
masks_per_sentence_empty[len(item)] += 1

masks_per_sentence_empty
{0: 18, 1: 591, 2: 97, 3: 34, 4: 15, 5: 12, 6: 5, 7: 2, 10: 3, 11: 1, 13: 1, 15: 1}
masks_empty_each = {}
for gap_list in sentences_empty_gaps:
    if len(gap_list) > 0:
        for gap in gap_list:
            masks_empty_each.setdefault(len(gap), 0)
            masks_empty_each[len(gap)] += 1

masks_empty_each
{3: 285, 1: 401, 6: 40, 2: 145, 8: 23, 4: 69, 12: 8, 10: 19, 14: 4, 5: 65, 9: 18, 20: 3, 22: 2, 13: 4, 27: 1, 18: 2,
 7: 19, 15: 4, 36: 1, 11: 4, 16: 1, 26: 1, 49: 1}
sorted_masks_empty_each = dict(sorted(masks_empty_each.items()))
sorted_masks_empty_each
{1: 401, 2: 145, 3: 285, 4: 69, 5: 65, 6: 40, 7: 19, 8: 23, 9: 18, 10: 19, 11: 4, 12: 8, 13: 4, 14: 4, 15: 4, 16: 1,
 18: 2, 20: 3, 22: 2, 26: 1, 27: 1, 36: 1, 49: 1}

masks_gaps_each = {}
for gap_list in sentences_masked_gaps:
    if len(gap_list) > 0:
        for gap in gap_list:
            masks_gaps_each.setdefault(len(gap), 0)
            masks_gaps_each[len(gap)] += 1

sorted_masks_gaps_each = dict(sorted(masks_gaps_each.items()))
sorted_masks_gaps_each
{1: 704, 2: 328, 3: 185, 4: 89, 5: 49, 6: 40, 7: 21, 8: 19, 9: 9, 10: 9, 11: 1, 12: 2, 13: 1, 16: 2, 18: 1, 19: 1,
 20: 1, 21: 1, 24: 1, 28: 1, 30: 1, 39: 1, 44: 1, 48: 1, 83: 1}

gap_total_masks = 0
gap_count_masks = 0
for key, value in sorted_masks_gaps_each.items():
    gap_total_masks += key * value
    gap_count_masks += value

gap_total_empty = 0
gap_count_empty = 0
for key, value in sorted_masks_empty_each.items():
    gap_total_empty += key * value
    gap_count_empty += value
