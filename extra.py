from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input="/home/miracle/Desktop/projects/facenobject/models/insightface.onnx",
    model_output="/home/miracle/Desktop/projects/facenobject/models/insightface_int8.onnx",
    weight_type=QuantType.QInt8
)


print("Quantized model saved as model_int8.onnx")
