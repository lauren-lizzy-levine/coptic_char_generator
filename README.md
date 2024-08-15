# coptic_char_generator

Repository for character based generation of Coptic for reconstruction of Coptic manuscripts.

## Data Path

Data comes from the [Coptic SCRIPTORIUM GitHub](https://github.com/CopticScriptorium/corpora). The recommended file path
is to clone the data repo into your Desktop, unzip the `sahidic.ot` and `sahidica.nt` files, and the data path will work without any updates or changes.

## Command Line Arguments
To run the project, the command is just `python main.py`

### Masking 
#### Mask Type (required)
Masking can be random (per character, by masking percentage) or smart (sections of masking, based on distribution of the text). 

Add `-m <random, smart>` or `--masking <random, smart>` to the command.

#### Masking Strategy (required)
Masking can happen only once (right after data is read in) or dynamic (re-masked in each training epoch). 

Add `-ms <once, dynamic>` or `--masking-strategy <once, dynamic>` to the command.

### SentencePiece (optional)

To train a SentencePiece model, add `-sp` or `--sentencepiece` to the command.

If you already have a SentencePiece model named "coptic_sp.model" and "coptic_sp.vocab" and don't need to retrain, leave
out the `-sp` flag. 

### Model Training (optional)

To train the model, add `-tr` or `--train` to the command. 

### Partition (optional)

To create the data set partition, add `-p` or `--partion` to the command.

### Evaluation (optional)

To evaluate on the lacuna test sets, add `-e` or `--eval` to the command.

## Demo

A demo for interacting with several of our models is available online [here](https://gucorpling.org/lacuna-demo?).
