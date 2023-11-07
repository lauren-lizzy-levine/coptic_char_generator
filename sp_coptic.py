import glob
import sentencepiece as spm

from coptic_utils import *

file_dir_path = f"{get_home_path()}/Desktop/corpora_tt"
files = ','.join(glob.glob(f"{file_dir_path}/*/*.tt")) # can use ',' separated list
logger.debug(files)
# note - this is currently running on all .tt files, including all the tags that we aren't interested in
# get this warning: trainer_interface.cc(122) LOG(WARNING) Too many sentences are loaded! (7535610), which may slow down training.


vocab_size = 1000
model_name = "coptic_sp"

# for a complete list of options see: https://github.com/google/sentencepiece/blob/master/doc/options.md
params = ""
params = params + " --input=" + files  # only required parameters
params = params + f" --model_prefix=./models/{model_name}"  # specify an output name

# optional parameters
params = params + " --vocab_size=" + str(vocab_size)  # default: 8000
params = params + ' --character_coverage=1.0'
params = params + " --model_type=char"

# params = params + ' --pad_id=3'					# include <pad> control symbol
# params = params + ' --control_symbols=<mask>,<oov>'

# trains a vocabulary and write 2 files: ./models/coptic_sp.model and ./models/coptic_sp.vocab
train = True
# train = False
if train:
    spm.SentencePieceTrainer.Train(params)

# load and test of model
fn_model = f"./models/{model_name}.model"

sp = spm.SentencePieceProcessor()
sp.Load(fn_model)

print(sp.__dict__)
print(sp.this)

print(sp.EncodeAsPieces("Hello world."))
print(sp.EncodeAsIds("Hello world."))

print()
print(sp.EncodeAsPieces("ⲕⲱⲧⲉ"))
print(sp.EncodeAsIds("ⲕⲱⲧⲉ"))

print()
print(sp.DecodeIds([10, 30, 60, 100]))  # just some random tokens
