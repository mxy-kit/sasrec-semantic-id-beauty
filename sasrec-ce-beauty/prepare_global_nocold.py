import torch
import torch.nn as nn


class SASRecCE(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super().__init__()

        self.usernum = usernum
        self.itemnum = itemnum
        self.dev = args.device
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units

        self.item_emb = nn.Embedding(itemnum + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(args.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_units,
            nhead=args.num_heads,
            dim_feedforward=args.hidden_units * 4,
            dropout=args.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=getattr(args, "norm_first", False),
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_blocks,
        )

        self.out = nn.Linear(args.hidden_units, itemnum + 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out.bias)

        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)
            self.pos_emb.weight[0].fill_(0)

    def _causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, seq):
        """
        seq: LongTensor, shape = (batch_size, maxlen)
        return logits over all items, shape = (batch_size, maxlen, itemnum + 1)
        """
        device = seq.device
        batch_size, seq_len = seq.shape

        positions = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        positions = positions * (seq != 0).long()

        x = self.item_emb(seq) * (self.hidden_units ** 0.5)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        padding_mask = seq.eq(0)
        causal_mask = self._causal_mask(seq_len, device)

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        logits = self.out(x)
        logits[:, :, 0] = -1e9
        return logits

    @torch.no_grad()
    def predict(self, seq):
        """
        seq: LongTensor, shape = (batch_size, maxlen)
        return logits for the last non-padding position.
        """
        logits = self.forward(seq)

        lengths = (seq != 0).sum(dim=1)
        last_pos = torch.clamp(lengths - 1, min=0)

        batch_idx = torch.arange(seq.size(0), device=seq.device)
        return logits[batch_idx, last_pos]