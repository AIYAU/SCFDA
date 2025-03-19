import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFeatureMOE(nn.Module):
    def __init__(self, feature_dim, num_features, num_experts, expert_hidden_dim, output_dim):
        """
        Args:
            feature_dim (int): Input dimension of each feature.
            num_features (int): Number of different features.
            num_experts (int): Number of expert networks.
            expert_hidden_dim (int): Hidden layer size of each expert.
            output_dim (int): Output dimension after feature fusion.
        """
        super(MultiFeatureMOE, self).__init__()

        # Experts for each feature
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(expert_hidden_dim, output_dim)
                )
                for _ in range(num_experts)
            ])
            for _ in range(num_features)
        ])

        # Gate Network: One gate for each feature
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, num_experts),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_features)
        ])

    def forward(self, features):
        """
        Args:
            features (list of Tensors): List of input feature tensors, each of shape (batch_size, feature_dim).
        Returns:
            fused_output (Tensor): Fused output tensor of shape (batch_size, output_dim).
        """
        batch_size = features[0].shape[0]
        num_features = len(features)
        output_dim = self.experts[0][0][-1].out_features  # Output dimension of each expert

        # Collect weighted outputs for each feature
        feature_outputs = []
        for i, feature in enumerate(features):
            # Get gate weights for current feature
            expert_weights = self.gates[i](feature)  # Shape: (batch_size, num_experts)
            # Get outputs from all experts for the current feature
            expert_outputs = torch.stack([expert(feature) for expert in self.experts[i]],dim=1)  # Shape: (batch_size, num_experts, output_dim)
            # Weighted sum of expert outputs
            weighted_output = torch.einsum('bn,bno->bo', expert_weights,expert_outputs)  # Shape: (batch_size, output_dim)
            feature_outputs.append(weighted_output)
        # Fuse outputs from all features (e.g., sum, mean, or concatenation)
        fused_output = torch.mean(torch.stack(feature_outputs, dim=0), dim=0)  # Shape: (batch_size, output_dim)
        return fused_output


# Example Usage
if __name__ == "__main__":
    batch_size = 16
    # Instantiate the MOE model
    moe = MultiFeatureMOE(feature_dim = 128, num_features = 9, num_experts = 4, expert_hidden_dim = 64, output_dim = 128)
    # Example input: 5 different features, each of shape (16, 128)
    features = []
    for _ in range(9):
        features.append(torch.randn(batch_size, 128))

    # print(torch.tensor(features).shape)
    fused_output = moe(features)
    print("Fused Output Shape:", fused_output.shape)  # Should print: (16, 128)
