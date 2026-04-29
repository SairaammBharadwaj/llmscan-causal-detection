import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelLoader:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,        # ✅ IMPORTANT
            output_hidden_states=True      # ✅ IMPORTANT
        )

        self.model.to(self.device)
        self.model.eval()

    def get_model(self):
        return self.model, self.tokenizer, self.device