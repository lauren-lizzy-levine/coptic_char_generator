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
        help="mask type: random, smart",
        choices=["random", "smart"],
        action="store",
    )
    parser.add_argument(
        "-ms",
        "--masking-strategy",
        required=True,
        help="masking strategy: once, dynamic",
        choices=["once", "dynamic"],
        action="store",
    )
    parser.add_argument(
        "-p",
        "--partition",
        required=False,
        help="create the data set partition",
        action="store_true",
    )
    parser.add_argument(
        "-ev",
        "--eval",
        required=False,
        help="conduct evaluation for actual lacuna test sets",
        action="store_true",
    )
    parser.add_argument(
        "-pr",
        "--predict",
        required=False,
        help="sentence to predict",
        action="store",
    )
    parser.add_argument(
        "-prk",
        "--predict_top_k",
        required=False,
        help="sentence to predict",
        action="store",
    )
    parser.add_argument(
        "-r",
        "--rank",
        required=False,
        help="ranking likelihood of options",
        action="store_true",
    )
    args = parser.parse_args()

    logger.info(f"start coptic data processing -- {datetime.datetime.now()}")

    # step 1 - set data files, partition the in data if needed (train/dev/test split)
    train_csv = "train.csv"
    dev_csv = "dev.csv"
    test_csv = "test.csv"
    empty_lacuna_csv = "test_empty_lacuna.csv"
    filled_reconstructed_lacuna_csv = "filled_test_reconstructed_lacuna.csv"
    masked_reconstructed_lacuna_csv = "masked_test_reconstructed_lacuna.csv"
    full_csv = "full_data.csv"

    if args.partition:
        # Do full data partition
        file_dir_path = f"{get_home_path()}/Desktop/corpora_tt/"
        file_list = glob.glob(f"{file_dir_path}*/*/*.tt")
        file_string = ",".join(file_list)
        logging.info(f"Files found: {len(file_list)}")
        (
            train_sentences,
            dev_sentences,
            test_sentences,
            empty_lacuna_sentences,
            filled_reconstructed_lacuna_sentences,
            masked_reconstructed_lacuna_sentences,
        ) = coptic_char_data.read_datafiles(file_list)
        full_data = train_sentences + dev_sentences + test_sentences
        logger.info(f"train: {len(train_sentences)} sentences")
        logger.info(f"dev: {len(dev_sentences)} sentences")
        logger.info(f"test: {len(test_sentences)} sentences")
        logger.info(f"full: {len(full_data)} sentences")
        logger.info(f"empty test: {len(empty_lacuna_sentences)} sentences")
        logger.info(f"recon test: {len(filled_reconstructed_lacuna_csv)} sentences")

        # write to partition files
        coptic_char_data.write_to_csv(train_csv, train_sentences)
        coptic_char_data.write_to_csv(dev_csv, dev_sentences)
        coptic_char_data.write_to_csv(test_csv, test_sentences)
        coptic_char_data.write_to_csv(full_csv, full_data)
        coptic_char_data.write_to_csv(empty_lacuna_csv, empty_lacuna_sentences)
        coptic_char_data.write_to_csv(
            filled_reconstructed_lacuna_csv, filled_reconstructed_lacuna_sentences
        )
        coptic_char_data.write_to_csv(
            masked_reconstructed_lacuna_csv, masked_reconstructed_lacuna_sentences
        )

    model_name = "coptic_smart_once_april_best"#"coptic_random_dynamic_5_13"#"coptic_sp_smart_dynamic"
    #model_name = "coptic_random_dynamic_5_13"
    # step 2 - sentence piece (on training)
    if args.sentencepiece:
        sp_coptic.create_sentencepiece_model(
            f"./data/{full_csv}", f"{model_name}", vocab_size=1000, train=True
        )

    # step 3 - model training
    mask_type = args.masking
    masking_strategy = args.masking_strategy

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
        #dev_data, mask = mask_input(model, dev_csv, mask_type, "once")
        #dev_list = [i for i in range(len(dev_data))]
        #accuracy_evaluation(model, dev_data, dev_list)
        #baseline_accuracy(model, dev_data, dev_list)

    logger.info(model)
    count_parameters(model)

    if args.train:
        training_data, mask = mask_input(model, train_csv, mask_type, masking_strategy)
        dev_data, _ = mask_input(model, dev_csv, mask_type, "once")

        model = train_model(
            model,
            training_data,
            dev_data=dev_data,
            output_name=model_name,
            mask=mask,
            mask_type=mask_type,
        )

    if args.eval:
        # run model on test set
        random_test_data, _ = mask_input(model, test_csv, "random", "once")
        random_test_list = [i for i in range(len(random_test_data))]
        smart_test_data, _ = mask_input(model, test_csv, "smart", "once")
        smart_test_list = [i for i in range(len(smart_test_data))]

        logging.info("Test Random:")
        accuracy_evaluation(model, random_test_data, random_test_list)
        baseline_accuracy(model, random_test_data, random_test_list)
        logging.info("Test Smart:")
        accuracy_evaluation(model, smart_test_data, smart_test_list)
        baseline_accuracy(model, smart_test_data, smart_test_list)

        # load sentences
        filled_lacuna_data = []
        file_path = f"./" + filled_reconstructed_lacuna_csv
        filled_lacuna_data = read_lacuna_test_files(file_path, filled_lacuna_data)
        logger.info(
            f"File {filled_reconstructed_lacuna_csv} read in with {len(filled_lacuna_data)} lines"
        )

        masked_lacuna_data = []
        file_path = f"./" + masked_reconstructed_lacuna_csv
        masked_lacuna_data = read_lacuna_test_files(file_path, masked_lacuna_data)
        logger.info(
            f"File {masked_reconstructed_lacuna_csv} read in with {len(masked_lacuna_data)} lines"
        )
        # make data_items
        reconstructed_data = []
        for i in range(len(masked_lacuna_data)):
            data_item = model.actual_lacuna_mask_and_label(
                DataItem(), masked_lacuna_data[i], filled_lacuna_data[i]
            )
            reconstructed_data.append(data_item)

        reconstructed_list = [i for i in range(len(reconstructed_data))]
        # accuracy evaluation
        logging.info("Test Reconstructed:")
        accuracy_evaluation(model, reconstructed_data, reconstructed_list)
        baseline_accuracy(model, reconstructed_data, reconstructed_list)

        # masked_empty_lacuna_csv = "test_empty_lacuna.csv"
        # # load sentences
        # empty_lacuna_data = []
        # file_path = f"./" + masked_empty_lacuna_csv
        # empty_lacuna_data = read_lacuna_test_files(file_path, empty_lacuna_data)
        # logger.info(
        #     f"File {masked_empty_lacuna_csv} read in with {len(empty_lacuna_data)} lines"
        # )
        # make data_items
        # empty_lacuna_data_items = []
        # for sentence in empty_lacuna_data:
        #     data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
        #     empty_lacuna_data_items.append(data_item)
        # # prediction
        # for i in range(len(empty_lacuna_data_items)):
        #     predict(model, empty_lacuna_data_items[i])
        #     if i >= 5:
        #         break

    if args.predict:
        sentence = args.predict
        data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
        predict(model, data_item)

    if args.predict_top_k:
        k = 1000
        sentence = args.predict_top_k
        data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
        predict_top_k(model, data_item, k)

    if args.rank:
        #sentence = "ⲛⲛⲉⲧⲛⲟⲩⲱϣϥⲛⲟⲩⲕⲁⲥⲉⲃⲟⲗⲛϩⲏⲧϥⲉⲕⲉⲁⲥⲡⲉϫⲁϥⲛⲧⲉ###ⲛⲟⲩⲟⲩϣⲏⲛⲟⲩⲱⲧⲉⲧⲉⲧⲛⲉⲟⲩⲟⲙϥⲕⲁⲧⲁⲛⲉⲧⲙⲡⲁⲧⲣⲓⲁⲙⲛⲛⲉⲧⲛⲇⲏⲙⲟⲥ"
        #options = ["ⲩⲉⲓ", "ⲓϩⲉ", "ⲧⲉⲓ", "ⲉⲉⲉ", "ⲁⲁⲗ"]
        #sentence = "ⲁⲕⲛⲟϭⲛⲉϭⲡϫⲟⲉⲓⲥⲁⲕϫⲟⲟⲥϫⲉϩⲙⲡⲁϣⲁⲓⲛⲛϩⲁⲣⲙⲁϯ#####ⲉϩⲣⲁⲓⲉⲡϫⲓⲥⲉ"
        #options = ["ⲟⲁⲟⲟⲓ", "ⲛⲁⲟⲩⲉ", "ⲛⲁⲁⲗⲉ", "ⲙⲟⲟϣⲉ"]
        #options = ["ⲛⲁⲃⲱⲕ", "ⲛⲁⲁⲗⲉ", "ⲙⲟⲟϣⲉ"]
        sentence = "ⲁⲥⲡⲁⲍⲉⲙⲙⲟⲥⲁⲧⲉⲥ#####ⲛϩⲁϩⲛⲥⲟⲡ"
        #sentence = "ⲁⲥⲡⲁⲍⲉⲙⲙⲟⲥⲉⲧⲉⲥ#####ⲛϩⲁϩⲛⲥⲟⲡ"
        options = ["ϩⲏⲩⲉⲛ", "ⲧⲁⲡⲣⲟ", "ⲡⲁⲓϭⲉ", "ⲟⲩⲟϭⲉ", "ϭⲁⲗⲟϫ", "ⲧⲉϩⲛⲉ", "ϩⲟⲟⲉⲉ"]
        char_indexes = [ind for ind, ele in enumerate(sentence) if ele == "#"]
        ranking = rank(model, sentence, options, char_indexes)
        print("Ranking:")
        print("(option, log_sum)")
        for option in ranking:
            print(option)

    logger.info(f"end generator -- {datetime.datetime.now()}\n")

