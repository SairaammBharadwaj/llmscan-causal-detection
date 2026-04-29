import torch

class LayerIntervention:
    def __init__(self, model):
        self.model = model

    def get_logit(self, logits):
        # Take first token logit (as paper does)
        return logits[0, -1, :].max().item()

    def run(self, input_ids):
        with torch.no_grad():
            original_outputs = self.model(
                input_ids=input_ids,
                output_attentions=False,
                output_hidden_states=False
            )

        original_logit = self.get_logit(original_outputs.logits)

        num_layers = len(self.model.transformer.h)

        selected_layers = [
            0,
            num_layers // 2,
            num_layers - 1
        ]

        layer_scores = []

        for skip_layer in selected_layers:

            hidden_states = self.model.transformer.wte(input_ids)
            hidden_states = self.model.transformer.drop(hidden_states)

            for i, block in enumerate(self.model.transformer.h):
                if i == skip_layer:
                    continue

                hidden_states = block(hidden_states)[0]

            hidden_states = self.model.transformer.ln_f(hidden_states)
            logits = self.model.lm_head(hidden_states)

            new_logit = self.get_logit(logits)

            ce = abs(original_logit - new_logit)
            layer_scores.append(ce)

        return layer_scores