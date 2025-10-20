from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="det_w600k_r50g.onnx",         # or w600k_r50.onnx
    model_output="det_w600k_r50g_int8.onnx",
    weight_type=QuantType.QInt8
)
