import torch

class AttentionExtractor:
    def __init__(self):
        pass

    def select_layers_heads(self, attentions):
        num_layers = len(attentions)

        layer_indices = [
            0,
            num_layers // 2,
            num_layers - 1
        ]

        selected = []

        for l in layer_indices:
            attn = attentions[l]  # [1, heads, seq, seq]
            if attn is None:
                continue
            num_heads = attn.shape[1]

            head_indices = [
                0,
                num_heads // 2,
                num_heads - 1
            ]

            selected_heads = attn[:, head_indices, :, :]
            selected.append(selected_heads)

        return selected