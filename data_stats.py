from collections import Counter


def char_histogram(file_name):
    with open(file_name, "r") as f:
        file_text = f.read()
        file_text = file_text.strip()
    char_counts = Counter(file_text)
    return char_counts


if __name__ == "__main__":
    char_histo = char_histogram("full_data.csv")
    print(char_histo)
