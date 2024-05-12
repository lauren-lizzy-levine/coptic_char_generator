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
        self.user_mask = sentence_piece.piece_to_id("#")

        self.num_tokens = sentence_piece.get_piece_size()
        self.specs = specs + [self.num_tokens]

        (
            embed_size,
            hidden_size,
            proj_size,
            rnn_nLayers,
            self.share,
            dropout,
            masking_proportion,
        ) = specs

        self.embed = nn.Embedding(self.num_tokens, embed_size)
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
        else:
            self.scale_down = nn.Linear(hidden_size, embed_size)

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

        if not self.share:
            output = self.out(output)
            output = torch.matmul(output, torch.t(self.embed.weight)) # this was added as a fix
        else:
            # use embedding table as output layer
            output = self.scale_down(output)
            output = torch.matmul(output, torch.t(self.embed.weight))

        return output

    def lookup_indexes(self, text, add_control=True):
        indexes = self.sentence_piece.EncodeAsIds(text)
        if add_control:
            indexes = [self.bos_id] + indexes + [self.eos_id]
        return indexes

    def decode(self, indexes):
        #print(indexes)
        tokens = self.sentence_piece.decode(indexes)
        return tokens

    def mask_and_label_characters(self, data_item, mask_type="random"):
        data_item.indexes = self.lookup_indexes(data_item.text)

        sentence_length = len(data_item.indexes)
        mask = [True] * sentence_length
        labels = [-100] * sentence_length

        mask_count = 0
        random_sub = 0
        orig_token = 0

        if mask_type == "random":
            for i in range(sentence_length):
                current_token = data_item.indexes[i]
                r_mask_status = random.random()
                r_mask_type = random.random()

                if r_mask_status < self.masking_proportion:
                    if r_mask_type < 0.8:
                        # replace with MASK symbol
                        replacement = self.mask
                        mask_count += 1
                    elif r_mask_type < 0.9:
                        # replace with random character
                        replacement = random.randint(3, self.num_tokens - 1)
                        random_sub += 1
                    else:
                        # retain original
                        replacement = current_token
                        orig_token += 1

                    data_item.indexes[i] = replacement
                    labels[i] = current_token

                else:
                    mask[i] = False

                data_item.mask = mask
                data_item.labels = labels

        elif mask_type == "smart":
            r_mask_quantity = random.randint(1, 5)

            mask_index = [0] * sentence_length
            i = 0

            while i < r_mask_quantity:
                r_start_loc = random.randint(0, sentence_length)
                r_mask_length = random.random()
                if r_mask_length <= 0.48:
                    mask_length = 1
                elif 0.48 < r_mask_length <= 0.70:
                    mask_length = 2
                elif 0.70 < r_mask_length <= 0.82:
                    mask_length = 3
                else:
                    mask_length = random.randint(4, 35)
                mask_end = r_start_loc + mask_length
                mask_type = random.random()
                mask_index[r_start_loc:mask_end] = [mask_type] * mask_length
                i += 1

            mask_start = 0
            for i in range(sentence_length):
                current_token = data_item.indexes[i]
                if mask_index[i] > 0:
                    if mask_index[i] < 0.8:
                        # replace with MASK symbol
                        replacement = self.mask
                        mask_count += 1
                    elif mask_index[i] < 0.9:
                        # replace with random character
                        replacement = random.randint(3, self.num_tokens - 1)
                        random_sub += 1
                    else:
                        # retain original
                        replacement = current_token
                        orig_token += 1

                    data_item.indexes[i] = replacement
                    labels[i] = current_token

                    mask_start += 1
                else:
                    mask[i] = False

                data_item.mask = mask
                data_item.labels = labels

        total_mask = mask_count + random_sub + orig_token

        return data_item, total_mask

    def actual_lacuna_mask_and_label(
        self, data_item, masked_sentence, filled_sentence=None
    ):
        if filled_sentence:
            data_item.text = filled_sentence
            filled_indexes = self.lookup_indexes(data_item.text)
        else:
            data_item.text = masked_sentence

        data_item.indexes = self.lookup_indexes(masked_sentence)

        sentence_length = len(data_item.indexes)
        mask = [False] * sentence_length
        labels = [-100] * sentence_length

        for i in range(len(data_item.indexes)):
            if data_item.indexes[i] == self.user_mask:
                data_item.indexes[i] = self.mask
                mask[i] = True
                if filled_sentence:
                    labels[i] = filled_indexes[i]
        data_item.mask = mask
        data_item.labels = labels

        return data_item
