import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("policy.onnx")

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

# Create input vector of 25 values
input_data = np.array([0.051034681598021, 0.0062437811756907774, 0.054983623865775981, -0.0026060565551273215, -0.065697319143097468, -0.25495819173794615, 0.0029001824270962918, -0.16818661232177423, -0.0030661369463357013, -0.011984224905356531, 0.0029960562263377233, 0.0062482933465453361, 0.015081175165340688, 0.0060926179838716177, 0.00898816867901729, 0.0029960562263342538, 0.16476936052428215, 0.048463764431206431, -0.50395160913467407, 1.5838474035263062, -0.1702549545288079, -0.0015661121578887105, 0.0069720675237476826, 0.0021370123140513897, 0.5], dtype=np.float32).reshape(1, -1)

print(f"\nInput shape: {input_data.shape}")
print(f"Input data: {input_data}")

# Run inference
output = session.run([output_name], {input_name: input_data})

print(f"\nOutput shape: {np.array(output).shape}")
print(f"Output: {output[0]}")
