import numpy as np

class FeatureExtractor:
    def extract(self, token_scores, layer_scores):
        token_scores = np.array(token_scores)

        features = [
            token_scores.mean(),
            token_scores.std(),
            token_scores.max(),
            token_scores.min(),
            token_scores.max() - token_scores.min()
        ]

        features.extend(layer_scores)

        return features