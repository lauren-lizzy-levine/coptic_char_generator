import argparse
import datetime
import glob

import coptic_char_data
import sp_coptic
from coptic_char_generator import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coptic character level generator")
    parser.add_argument(
        "-tr",
        "--train",
        required=False,
        help="True to train the model",
        action="store_true",
    )
    parser.add_argument(
        "-sp",
        "--sentencepiece",
        required=False,
        help="true to skip sentencepiece model training",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--masking",
        required=True,
        help="masking strategy: random or smart",
        action="store",
    )
    parser.add_argument(
        "-p",
        "--partition",
        required=False,
        help="create the data set partition",
        action="store_true",
    )
    args = parser.parse_args()

    logger.info(f"start coptic data processing -- {datetime.datetime.now()}")

    # step 1 - set data files, partition the in data if needed (train/dev/test split)
    train_csv = "train.csv"
    dev_csv = "dev.csv"
    test_csv = "test.csv"
    empty_lacuna_csv = "test_empty_lacuna.csv"
    reconstructed_lacuna_csv = "test_reconstructed_lacuna.csv"
    full_csv = "full_data.csv"

    if args.partition:
        # Do full data partition
        file_dir_path = f"{get_home_path()}/Desktop/corpora_tt/"
        file_list = glob.glob(f"{file_dir_path}*/*/*.tt")
        file_string = ",".join(file_list)
        logging.info(f"Files found: {len(file_list)}")
        train_sentences, dev_sentences, test_sentences, \
            empty_lacuna_sentences, reconstructed_lacuna_sentences = coptic_char_data.read_datafiles(file_list)
        full_data = train_sentences + dev_sentences + test_sentences
        logger.info(f"train: {len(train_sentences)} sentences")
        logger.info(f"dev: {len(dev_sentences)} sentences")
        logger.info(f"test: {len(test_sentences)} sentences")
        logger.info(f"full: {len(full_data)} sentences")
        logger.info(f"empty test: {len(empty_lacuna_sentences)} sentences")
        logger.info(f"recon test: {len(reconstructed_lacuna_sentences)} sentences")

        # write to partition files
        coptic_char_data.write_to_csv(train_csv, train_sentences)
        coptic_char_data.write_to_csv(dev_csv, dev_sentences)
        coptic_char_data.write_to_csv(test_csv, test_sentences)
        coptic_char_data.write_to_csv(full_csv, full_data)
        coptic_char_data.write_to_csv(empty_lacuna_csv, empty_lacuna_sentences)
        coptic_char_data.write_to_csv(reconstructed_lacuna_csv, reconstructed_lacuna_sentences)

    # TODO - what other information might we want to in the csv?

    # csv_name = "english.csv"
    model_name = "coptic_sp"

    # TODO - masking options here - mask before we create sentencepiece model
    logger.info(f"Masking type: {args.masking}")
    # random - masking as we have right now
    # smart - smart masking based on the text

    # step 2 - sentence piece (on training)
    if args.sentencepiece:
        sp_coptic.create_sentencepiece_model(
            full_csv, f"{model_name}", vocab_size=1000, train=True
        )

    # step 3 - model training
    if args.train:
        logger.info("Training a sentencepiece model")
        sp = sp_coptic.spm.SentencePieceProcessor()
        sp.Load(model_path + model_name + ".model")

        logger.info(
            f"Train {model_name} model specs: embed_size: {specs[0]}, hidden_size: {specs[1]}, proj_size: {specs[2]}, rnn n layers: {specs[3]}, share: {specs[4]}, dropout: {specs[5]}"
        )

        model = RNN(sp, specs)
        model.tokenizer = model_name
        model = model.to(device)
    else:
        logger.info(f"Using a pre-trained model")
        model = torch.load(model_path + model_name + ".pth", map_location=device)
        logger.info(
            f"Load model: {model} with specs: embed_size: {model.specs[0]}, hidden_size: {model.specs[1]}, proj_size: {model.specs[2]}, rnn n layers: {model.specs[3]}, share: {model.specs[4]}, dropout: {model.specs[5]}"
        )
        # Eval on Dev data
        dev_data = []
        file_path = f"./" + dev_csv
        dev_data = read_datafile(file_path, dev_data)
        dev_list = [i for i in range(len(dev_data))]
        fixed_dev_data = []
        for data_item in dev_data:
            masked_data_item, _ = model.mask_and_label_characters(data_item)
            fixed_dev_data.append(masked_data_item)
        accuracy_evaluation(model, fixed_dev_data, dev_list)
        baseline_accuracy(fixed_dev_data, dev_list)

    logger.info(model)
    count_parameters(model)

    if args.train:
        data = []
        file_path = f"./" + train_csv
        data = read_datafile(file_path, data)
        logger.info(f"File {train_csv} read in with {len(data)} lines")
        fixed_data = []
        for data_item in data:
            masked_data_item, _ = model.mask_and_label_characters(data_item)
            fixed_data.append(masked_data_item)

        model = train_model(
            model, fixed_data, dev_data=fixed_data, output_name=model_name
        )

    logger.info(f"end generator -- {datetime.datetime.now()}\n")
