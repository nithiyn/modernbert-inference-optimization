# compile_torch_compile_trt.py
# Dynamic Just-In-Time compilation using torch.compile + TensorRT backend
# Allows hybrid fallback and FlashAttention2 kernels.

import torch
import torch_tensorrt
from transformers import AutoModelForSequenceClassification
import time
import warnings
warnings.filterwarnings("ignore")


def compile_with_torch_compile_trt(checkpoint_path, batch_sizes, max_lengths):
    """
    Uses torch.compile with TensorRT backend for runtime optimization.
    - Dynamic shape support enabled (no explicit ranges needed)
    - Allows fallback to PyTorch ops automatically
    - Works with FlashAttention2 backend
    """
    print("=== torch.compile + TensorRT Backend (Dynamic Shapes) ===")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        reference_compile=False,
        attn_implementation="flash_attention_2",  # Allowed here
    ).eval()

    # Compile model (runtime JIT)
    compiled_model = torch.compile(
        model,
        backend="torch_tensorrt",
        dynamic=True,
        options={
            "enabled_precisions": {torch.bfloat16},
            "workspace_size": 8 << 30,
            "min_block_size": 3,
            "assume_dynamic_shape_support": True,
            "optimization_level": 5,
            "debug": False,
        },
    )

    print("✅ torch.compile(TensorRT) done! Supports dynamic shapes + fallbacks.")
    return compiled_model


def benchmark_dynamic_shapes(compiled_model, test_configs):
    """Quick runtime sanity check across varying shapes."""
    print("\n=== Benchmarking dynamic shapes ===")
    for (b, s) in test_configs:
        x = torch.randint(0, 50368, (b, s), device="cuda")
        mask = torch.ones((b, s), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = compiled_model(input_ids=x, attention_mask=mask)
        torch.cuda.synchronize()
        print(f"Ran shape ({b}, {s}) in {(time.time() - t0)*1000:.2f} ms")


def main():
    checkpoint = "/home/ubuntu/bf16_checkpoints"
    batch_sizes = [4, 8, 16]
    max_lengths = [512, 1024, 2048, 4096]

    compiled = compile_with_torch_compile_trt(checkpoint, batch_sizes, max_lengths)
    test_shapes = [(4, 512), (8, 1024), (8, 2048)]
    benchmark_dynamic_shapes(compiled, test_shapes)


if __name__ == "__main__":
    main()
