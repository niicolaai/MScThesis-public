import onnx
from scipy.io import savemat
from pathlib import Path

# Directory containing ONNX models
models_dir = Path("models")
outputs_dir = Path("converted")

# Create outputs directory if it doesn't exist
outputs_dir.mkdir(exist_ok=True)

# Find all ONNX files in the models directory
onnx_files = list(models_dir.glob("*.onnx"))

if not onnx_files:
    print(f"No ONNX files found in {models_dir}")
else:
    print(f"Found {len(onnx_files)} ONNX file(s). Starting conversion...\n")

    for onnx_path in onnx_files:
        try:
            print(f"Converting {onnx_path.name}...", end=" ")

            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))

            # Extract weights
            weights_dict = {}
            for initializer in onnx_model.graph.initializer:
                tensor = onnx.numpy_helper.to_array(initializer)
                # Replace dots with underscores to make valid MATLAB variable names
                clean_name = initializer.name.replace('.', '_')
                weights_dict[clean_name] = tensor

            # Generate output filename
            output_filename = outputs_dir / f"converted_model_{onnx_path.stem}.mat"

            # Save to .mat file
            savemat(output_filename, weights_dict)
            print(f"✓ Saved to {output_filename}")

        except Exception as e:
            print(f"✗ Error: {e}")

    print("\nBulk conversion complete!")
