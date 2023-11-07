import argparse
import datetime
import glob

import coptic_char_data
import sp_coptic
from coptic_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coptic character level generator")
    parser.add_argument(
        "-m", "--model", required=False, help="Name of pre-trained model"
    )

    # step 1 - read in data
    logger.info(
        f"\nstart coptic data processing -- {datetime.datetime.now()}"
    )

    file_dir_path = f"{get_home_path()}/Desktop/corpora_tt/"
    file_list = glob.glob(f"{file_dir_path}*/*.tt")
    file_string = ','.join(file_list)
    logging.info(f"Files found: {len(file_list)}")
    sentences = coptic_char_data.read_datafiles(file_list)
    logger.info(f"File read: {len(sentences)} sentences")

    #TO DO - what other information might we want to in the csv?

    # step 2 - write to csv


    # step 3 - train/dev/test split (we may want to do this before step 1)

    # step 4 - sentence piece (on training)
    model_name = "coptic_sp"
    sp_coptic.create_sentencepiece_model(file_string, model_name, vocab_size=1000, train=True)
    # note - this is currently running on all .tt files, including all the tags that we aren't interested in
    # get this warning: trainer_interface.cc(122) LOG(WARNING) Too many sentences are loaded! (7535610), which may slow down training.


    # step 5 - model training