import torch

class InferenceEngine:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True
            )

        return {
            "input_ids": inputs["input_ids"],
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states,
            "logits": outputs.logits
        }