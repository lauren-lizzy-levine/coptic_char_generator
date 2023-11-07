import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, sentence_piece, specs):
        super(RNN, self).__init__()
        # ensure that character dictionary doesn't change
        self.sentence_piece = sentence_piece  # SentencePiece model
        self.unk_id = sentence_piece.piece_to_id("<unk>")
        self.bos_id = sentence_piece.piece_to_id("<s>")
        self.eos_id = sentence_piece.piece_to_id("</s>")

        nTokens = sentence_piece.get_piece_size()
        self.specs = specs + [nTokens]

        embed_size, hidden_size, proj_size, rnn_nLayers, self.share, dropout = specs
        self.embed = nn.Embedding(nTokens, embed_size)

        # if rnn_nLayers == 1: dropout = 0.0 # dropout is only applied between layers
        self.rnn = nn.LSTM(
            embed_size,
            hidden_size,
            rnn_nLayers,
            dropout=dropout,
            batch_first=True,
            proj_size=proj_size,
        )

        if not self.share:
            self.out = nn.Linear(
                proj_size, nTokens, bias=False
            )  # character - CrossEntropy

        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p)
                nn.init.kaiming_normal_(p)
                pass

    def forward(self, seqs, hidden=None):
        print(seqs)
        nBatch = len(seqs)
        nTokens = len(seqs[0])
        seqs = torch.cat(seqs).view(nBatch, nTokens)
        embed = self.embed(seqs)
        embed = self.dropout(embed)
        prev, hidden = self.rnn(embed, hidden)
        print(hidden[0].shape)
        print(hidden[1].shape)
        prev = self.dropout(prev)
        if not self.share:
            out = self.out(prev)  # chars
        else:
            out = torch.matmul(
                prev, torch.t(self.embed.weight)
            )  # uses the embedding table as the output layer

        return out, hidden

    def lookup_ndxs(self, text, add_control=True):
        indexes = self.sentence_piece.EncodeAsIds(text)
        if add_control:
            indexes = [self.bos_id] + indexes + [self.eos_id]
        return indexes

    def decode(self, indexes):
        tokens = self.sentence_piece.decode(indexes)
        return tokens
