import torch
import scipy.io as sio
import numpy as np
import os

# Configuration variables
input_model = "policy.onnx"
output_model = "2026_01_26_domain_randomization_model"

# Detect file extension and load accordingly
file_extension = os.path.splitext(input_model)[1].lower()

if file_extension == ".pt":
    # Load PyTorch checkpoint
    state_dict = torch.load(input_model, map_location="cpu")
    print(f"Loaded PyTorch model from {input_model}")
elif file_extension == ".onnx":
    # Load ONNX model and convert to PyTorch state_dict
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Error: onnx and onnxruntime are required for ONNX support.")
        print("Install them with: pip install onnx onnxruntime")
        exit(1)

    onnx_model = onnx.load(input_model)
    onnx.checker.check_model(onnx_model)

    # Convert ONNX to PyTorch state_dict
    state_dict = {}
    for initializer in onnx_model.graph.initializer:
        state_dict[initializer.name] = torch.from_numpy(np.array(onnx.numpy_helper.to_array(initializer)))

    print(f"Loaded ONNX model from {input_model}")
else:
    print(f"Error: Unsupported file format '{file_extension}'. Supported formats: .pt, .onnx")
    exit(1)

# Network architecture parameters
input_dim = 26
hidden_dims = [32, 16]
output_dim = 8
activation = "elu"


# Create actor network
class ActorNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="elu"):
        super().__init__()
        self.activation = torch.nn.ELU() if activation == "elu" else torch.nn.ReLU()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Instantiate actor model
model = ActorNetwork(input_dim, hidden_dims, output_dim, activation)

# Filter state_dict to get only actor weights
actor_state_dict = {}
for key, value in state_dict.items():
    if "actor" in key.lower():
        # Remove 'actor.' prefix if present
        new_key = key.replace("actor.", "") if "actor." in key else key
        actor_state_dict[new_key] = value

# If no actor prefix found, use all keys and add "network." prefix
if not actor_state_dict:
    actor_state_dict = state_dict.copy()

# Rename keys to match model structure (add "network." prefix)
renamed_state_dict = {}
for key, value in actor_state_dict.items():
    if not key.startswith("network."):
        renamed_state_dict[f"network.{key}"] = value
    else:
        renamed_state_dict[key] = value

# Load the weights into the model
if renamed_state_dict:
    model.load_state_dict(renamed_state_dict)
    print("Loaded actor weights from checkpoint")
else:
    print(
        "Warning: No actor weights found in checkpoint. Check key names in state_dict."
    )

model.eval()

# Create dummy input matching your observation dimension
dummy_input = torch.randn(1, input_dim)

# Trace the model
traced_model = torch.jit.trace(model, dummy_input)
print("Model traced successfully")

# Extract weights and structure
model_dict = {
    "model": traced_model,  # Save the traced model itself
    "state_dict": model.state_dict(),  # Save weights as dict
    "input_dim": input_dim,
}

# Convert state_dict weights to numpy for MATLAB compatibility
weights_numpy = {}
for name, param in model.state_dict().items():
    weights_numpy[name] = param.cpu().numpy()

# Save to .mat file (MATLAB format)
sio.savemat(f"{output_model}.mat", weights_numpy)
print(f"Model weights saved to {output_model}.mat")

# Also save the traced model for PyTorch use
traced_model.save(f"{output_model}_traced.pt")
print(f"Traced model saved to {output_model}_traced.pt")

# Optional: Print network structure
print("\nNetwork structure:")
print(model)
print("\nWeights saved:")
for name in weights_numpy.keys():
    print(f"  {name}: {weights_numpy[name].shape}")
