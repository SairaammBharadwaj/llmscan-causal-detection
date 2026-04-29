import torch

class TokenIntervention:
    def __init__(self, engine, extractor):
        self.engine = engine
        self.extractor = extractor

    def replace_token(self, input_ids, index, pad_token_id):
        tokens = input_ids.clone()
        tokens[0, index] = pad_token_id
        return tokens

    def compute_distance(self, attn1, attn2):
        dist = 0
        for a1, a2 in zip(attn1, attn2):
            # Ensure same shape
            min_len = min(a1.shape[-1], a2.shape[-1])

            a1 = a1[:, :, :min_len, :min_len]
            a2 = a2[:, :, :min_len, :min_len]

            dist += torch.norm(a1 - a2).item()
        return dist

    def run(self, prompt, tokenizer):
        # Original
        original = self.engine.run(prompt)
        original_attn = self.extractor.select_layers_heads(original["attentions"])

        input_ids = original["input_ids"]
        seq_len = input_ids.shape[1]

        pad_token_id = tokenizer.eos_token_id

        scores = []

        for i in range(seq_len):
            # Replace token directly
            modified_ids = self.replace_token(input_ids, i, pad_token_id)

            # Run model directly on IDs (NO decode)
            with torch.no_grad():
                outputs = self.engine.model(
                    input_ids=modified_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )

            modified_attn = self.extractor.select_layers_heads(outputs.attentions)

            dist = self.compute_distance(original_attn, modified_attn)
            scores.append(dist)

        return scores