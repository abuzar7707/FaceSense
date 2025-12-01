import onnxruntime as ort

session = ort.InferenceSession("emotion-ferplus-8.onnx")

print("Inputs:")
for i in session.get_inputs():
    print("  Name:", i.name)
    print("  Shape:", i.shape)
    print("  Type:", i.type)

print("\nOutputs:")
for o in session.get_outputs():
    print("  Name:", o.name)
    print("  Shape:", o.shape)
    print("  Type:", o.type)
