# coptic_char_generator

Repository for character based generation of Coptic for reconstruction of Coptic manuscripts.

## Data Path

Data comes from the [Coptic SCRIPTORIUM GitHub](https://github.com/CopticScriptorium/corpora). The recommended file path
is to clone the data repo into your Desktop and the data path will work without any updates or changes.

## Command Line Arguments

### SentencePiece

To run the project, the command is just `python main.py`

To train a SentencePiece model, add `-sp` or `-sentencepiece` to the command.

If you already have a SentencePiece model named "coptic_sp.model" and "coptic_sp.vocab" and don't need to retrain, leave
out the `-sp` flag. 

### Model Training

To train 