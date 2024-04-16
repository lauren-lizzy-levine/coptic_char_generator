import sentencepiece as spm

from coptic_utils import *


def create_sentencepiece_model(files, model_name, vocab_size=1000, train=True):
    # for a complete list of options see: https://github.com/google/sentencepiece/blob/master/doc/options.md
    params = ""
    params = params + " --input=" + files  # only required parameters
    params = params + f" --model_prefix=./models/{model_name}"  # specify an output name

    # optional parameters
    params = params + " --vocab_size=" + str(vocab_size)  # default: 8000
    params = params + " --character_coverage=1.0"
    params = params + " --model_type=char" # <unigram (default), bpe, char, word>

    # params = params + ' --pad_id=3'					# include <pad> control symbol
    params = params + " --control_symbols=<mask>"  # ,<oov>'
    params = params + " --user_defined_symbols=#"  # ,<oov>'

    # trains a vocabulary and write 2 files: ./models/coptic_sp.model and ./models/coptic_sp.vocab
    if train:
        spm.SentencePieceTrainer.Train(params)

    # load and test of model
    fn_model = f"./models/{model_name}.model"

    logger.info(f"starting SentencePiece {model_name} with {params}")

    sp = spm.SentencePieceProcessor()
    sp.Load(fn_model)

    logger.info(f"SentencePiece model {model_name} created")
    # logger.info(sp.EncodeAsIds("##"))

    # print(sp.__dict__)
    # print(sp.this)
    #
    # print(sp.EncodeAsPieces("Hello world."))
    # print(sp.EncodeAsIds("Hello world."))
    #
    # print()
    # print(sp.EncodeAsPieces("ⲕⲱⲧⲉ"))
    # print(sp.EncodeAsIds("ⲕⲱⲧⲉ"))
    # print(f"<mask> sp: {sp.PieceToId('<mask>')}")
    #
    # print(f"decode 0: {sp.DecodeIds(0)}")
    # print(f"decode 1: {sp.DecodeIds(1)}")
    # print(f"decode 2: {sp.DecodeIds(2)}")
    # print(f"decode 3: {sp.DecodeIds(3)}")
    # print(f"decode 4: {sp.DecodeIds(4)}")
    #
    # print()
    # print(sp.DecodeIds([10, 32, 45, 14]))  # just some random tokens
