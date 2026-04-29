from backend.model.model_loader import ModelLoader
from backend.model.inference import InferenceEngine
from backend.causality.attention_extractor import AttentionExtractor
from backend.causality.token_intervention import TokenIntervention
from backend.causality.layer_intervention import LayerIntervention
from backend.detector.feature_extractor import FeatureExtractor

def main():
    print("Loading model...")

    # Load model
    loader = ModelLoader()
    model, tokenizer, device = loader.get_model()

    # Initialize inference engine
    engine = InferenceEngine(model, tokenizer, device)

    # Initialize attention extractor
    extractor = AttentionExtractor()

    # Test prompt
    prompt = "What is the capital of France?"

    print("\nRunning inference...\n")

    # Run model
    result = engine.run(prompt)

    # Extract selected attention
    selected_attn = extractor.select_layers_heads(result["attentions"])

    # Print outputs
    print("===== BASIC INFO =====")
    print("Input IDs shape:", result["input_ids"].shape)
    print("Total layers:", len(result["attentions"]))
    print("Full attention shape (layer 0):", result["attentions"][0].shape)
    print("Logits shape:", result["logits"].shape)

    print("\n===== SELECTED ATTENTION =====")
    print("Selected layers:", len(selected_attn))
    print("Selected attention shape (per layer):", selected_attn[0].shape)

    token_intervention = TokenIntervention(engine, extractor)

    scores = token_intervention.run(prompt, tokenizer)

    print("\n===== TOKEN CAUSAL SCORES =====")
    print(scores)

    layer_intervention = LayerIntervention(model)

    layer_scores = layer_intervention.run(result["input_ids"])

    print("\n===== LAYER CAUSAL SCORES =====")
    print(layer_scores)

    feature_extractor = FeatureExtractor()

    features = feature_extractor.extract(scores, layer_scores)

    print("\n===== FEATURE VECTOR =====")
    print(features)

if __name__ == "__main__":
    main()