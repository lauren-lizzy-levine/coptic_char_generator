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

        self.scale_up = nn.Linear(embed_size, hidden_size)

        self.rnn = nn.LSTM(
            hidden_size,
            int(hidden_size / 2),
            num_layers=rnn_nLayers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        if not self.share:
            self.out = nn.Linear(hidden_size, embed_size)

        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p)
                nn.init.kaiming_normal_(p)
                pass

    def forward(self, seqs):
        num_batches = len(seqs)
        num_tokens = len(seqs[0])
        seqs = torch.cat(seqs).view(num_batches, num_tokens)

        embed = self.embed(seqs)
        embed = self.dropout(embed)
        embed = self.scale_up(embed)

        output, _ = self.rnn(embed)
        output = self.dropout(output)

        output = self.out(output)
        output = torch.matmul(output, torch.t(self.embed.weight))

        return output

    def lookup_indexes(self, text, add_control=True):
        indexes = self.sentence_piece.EncodeAsIds(text)
        if add_control:
            indexes = [self.bos_id] + indexes + [self.eos_id]
        return indexes

    def decode(self, indexes):
        tokens = self.sentence_piece.decode(indexes)
        return tokens

    def mask_and_label_characters(self, data_item):
        data_item.indexes = self.lookup_indexes(data_item.text)

        sentence_length = len(data_item.indexes)
        mask = [True] * sentence_length
        labels = [-100] * sentence_length

        mask_count = 0
        random_sub = 0
        orig_token = 0

        for i in range(len(data_item.indexes)):
            current_token = data_item.indexes[i]
            r1 = random.random()
            r2 = random.random()

            if r1 < self.masking_proportion:
                if r2 < 0.8:
                    # print("masked")
                    # replace with MASK symbol
                    replacement = self.mask
                    mask_count += 1
                elif r2 < 0.9:
                    # print("random replacement")
                    # replace with random character
                    replacement = random.randint(3, self.num_tokens - 1)
                    random_sub += 1
                else:
                    # print("orig")
                    # retain original
                    replacement = current_token
                    orig_token += 1

                data_item.indexes[i] = replacement
                labels[i] = current_token

                # print(f"current: {current_token}, replacement: {replacement}, data_item: {data_item.masked_indexes[i]}")

            else:
                mask[i] = False

            data_item.mask = mask
            data_item.labels = labels

        logger.debug(f"mask: {mask_count}, random: {random_sub}, orig: {orig_token}")
        total_mask = mask_count + random_sub + orig_token

        return data_item, total_mask
