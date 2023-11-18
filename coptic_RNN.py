import torch.nn as nn
import random

from coptic_utils import *

MASK = "<mask>"


class RNN(nn.Module):
    def __init__(self, sentence_piece, specs):
        super(RNN, self).__init__()
        # ensure that character dictionary doesn't change
        self.sentence_piece = sentence_piece  # SentencePiece model
        self.unk_id = sentence_piece.piece_to_id("<unk>")
        self.bos_id = sentence_piece.piece_to_id("<s>")
        self.eos_id = sentence_piece.piece_to_id("</s>")
        self.mask = sentence_piece.piece_to_id("<mask>")

        num_tokens = sentence_piece.get_piece_size()
        self.num_tokens = num_tokens
        self.specs = specs + [num_tokens]

        (
            embed_size,
            hidden_size,
            proj_size,
            rnn_nLayers,
            self.share,
            dropout,
            masking_proportion,
        ) = specs
        self.embed = nn.Embedding(num_tokens, embed_size)
        self.masking_proportion = masking_proportion

        # if rnn_nLayers == 1: dropout = 0.0 # dropout is only applied between layers
        self.rnn = nn.LSTM(
            embed_size,
            hidden_size,
            rnn_nLayers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
            proj_size=proj_size,
        )

        if not self.share:
            self.out = nn.Linear(proj_size * 2, num_tokens, bias=False)

        # TODO - figure out how to update input for share?

        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p)
                nn.init.kaiming_normal_(p)
                pass

    def forward(self, seqs, hidden=None):
        logger.debug(seqs)
        num_batches = len(seqs)
        num_tokens = len(seqs[0])
        seqs = torch.cat(seqs).view(num_batches, num_tokens)

        embed = self.embed(seqs)
        embed = self.dropout(embed)
        prev, hidden = self.rnn(embed, hidden)

        logger.debug(hidden[0].shape)
        logger.debug(hidden[1].shape)

        prev = self.dropout(prev)
        if not self.share:
            out = self.out(prev)  # chars
        else:
            out = torch.matmul(prev, torch.t(self.embed.weight))

        return out, hidden

    def lookup_indexes(self, text, add_control=True):
        indexes = self.sentence_piece.EncodeAsIds(text)
        if add_control:
            indexes = [self.bos_id] + indexes + [self.eos_id]
        return indexes

    def decode(self, indexes):
        tokens = self.sentence_piece.decode(indexes)
        return tokens

    def mask_and_label_characters(self, data_item):
        sentence_length = len(data_item.indexes)
        mask = [True] * sentence_length
        labels = [-100] * sentence_length

        for i in range(len(data_item.indexes)):
            current_token = data_item.indexes[i]
            r1 = random.random()
            r2 = random.random()

            if r1 < self.masking_proportion:
                if r2 < 0.8:
                    # replace with MASK symbol
                    replacement = self.mask
                elif r2 < 0.9:
                    # replace with random character
                    replacement = random.randint(3, self.num_tokens - 1)
                else:
                    # retain original
                    replacement = current_token

                data_item.indexes[i] = replacement
                labels[i] = current_token

            else:
                mask[i] = False

            data_item.mask = mask
            data_item.labels = labels

        return data_item
