import onnx
import numpy as np
from scipy.io import savemat

onnx_model = onnx.load("policy.onnx")

weights_dict = {}
for initializer in onnx_model.graph.initializer:
    tensor = onnx.numpy_helper.to_array(initializer)
    # Replace dots with underscores to make valid MATLAB variable names
    clean_name = initializer.name.replace('.', '_')
    weights_dict[clean_name] = tensor

savemat("converted_model.mat", weights_dict)
print("Successfully converted ONNX model to MATLAB .mat format.")
