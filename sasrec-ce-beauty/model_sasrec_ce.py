import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)
        return outputs


class SASRecCE(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRecCE, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.item_emb = torch.nn.Embedding(
            self.item_num + 1,
            args.hidden_units,
            padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1,
            args.hidden_units,
            padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )

            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    args.hidden_units,
                    args.num_heads,
                    args.dropout_rate
                )
            )

            self.forward_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )

            self.forward_layers.append(
                PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            )

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        poss = np.tile(
            np.arange(1, log_seqs.shape[1] + 1),
            [log_seqs.shape[0], 1]
        )
        poss *= (log_seqs != 0)

        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)

            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](
                    x, x, x,
                    attn_mask=attention_mask
                )
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](
                    self.forward_layernorms[i](seqs)
                )
            else:
                mha_outputs, _ = self.attention_layers[i](
                    seqs, seqs, seqs,
                    attn_mask=attention_mask
                )
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](
                    seqs + self.forward_layers[i](seqs)
                )

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, log_seqs):
        log_feats = self.log2feats(log_seqs)
        logits = torch.matmul(log_feats, self.item_emb.weight.transpose(0, 1))
        return logits

    def predict(self, log_seqs, item_indices=None):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]

        if item_indices is None:
            logits = torch.matmul(final_feat, self.item_emb.weight.transpose(0, 1))
            return logits

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
