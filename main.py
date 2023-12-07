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
    args = parser.parse_args()

    logger.info(f"start coptic data processing -- {datetime.datetime.now()}")

    # TODO - train/dev/test split (we may want to do this before step 1

    # step 1 - read in data
    # TODO - create a set that has all the file names to track any duplicates

    file_dir_path = f"{get_home_path()}/Desktop/corpora_tt/"
    file_list = glob.glob(f"{file_dir_path}*/*/*.tt")
    file_string = ",".join(file_list)
    logging.info(f"Files found: {len(file_list)}")
    sentences = coptic_char_data.read_datafiles(file_list)
    logger.info(f"Files read: {len(sentences)} sentences")

    # TODO - what other information might we want to in the csv?

    csv_name = "coptic_sentences.csv"
    model_name = "coptic_sp"

    # TODO - masking options here - mask before we create sentencepiece model
    logger.info(f"Masking type: {args.masking}")
    # random - masking as we have right now
    # smart - smart masking based on the text

    # step 2 - write to csv
    coptic_char_data.write_to_csv(csv_name, sentences, plain=True)

    # step 3 - sentence piece (on training)
    if args.sentencepiece:
        sp_coptic.create_sentencepiece_model(
            csv_name, f"{model_name}", vocab_size=1000, train=True
        )

    # step 4 - model training
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

    logger.info(model)
    count_parameters(model)

    if args.train:
        data = []
        file_path = f"./" + csv_name
        read_datafile(file_path, data)
        logger.info(f"File {csv_name} read in with {len(data)} lines")

        model = train_model(model, data, dev_data=data, output_name=model_name)

    logger.info(f"end generator -- {datetime.datetime.now()}\n")
