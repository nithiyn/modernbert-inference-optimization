# compile_aot_torch_tensorrt_dynamo.py
# Ahead-of-Time (AOT) TensorRT compilation with torch.export + torch_tensorrt.dynamo

import torch
import torch_tensorrt
from transformers import AutoModelForSequenceClassification

def compile_with_aot_torch_tensorrt_dynamo(checkpoint_path):
    """
    Exports the model to a static graph and compiles it ahead-of-time using TensorRT.
    - No graph breaks allowed (torch.export requirement)
    - Fallback to PyTorch is disabled (strict mode)
    - Dynamic shapes supported via explicit range specifications
    """
    print("=== AOT Compilation with torch_tensorrt.dynamo (torch.export) ===")

    # Load model with eager attention (FlashAttention2 not supported in export)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        reference_compile=False
    ).eval()

    # Dummy input for export (use dynamic dims)
    example_inputs = (
        torch.randint(0, 50368, (8, 1024), device="cuda"),
        torch.ones((8, 1024), dtype=torch.int64, device="cuda"),
    )

    # Export full static graph (AOT)
    print("Exporting model...")
    exported_program = torch.export.export(
        model,
        args=(example_inputs[0], example_inputs[1]),
        dynamic_shapes={
            "input_ids": {
                0: torch.export.Dim("batch", min=2, max=4),
                1: torch.export.Dim("seq", min=2048, max=4096)
            },
            "attention_mask": {
                0: torch.export.Dim("batch", min=2, max=4),
                1: torch.export.Dim("seq", min=2048, max=4096)
            },
        }
    )


    # Compile with TensorRT
    print("Compiling exported program with TensorRT (strict mode)...")
    trt_model = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[
            # input_ids
            torch_tensorrt.Input(
                min_shape=(2, 512),
                opt_shape=(4, 1024),
                max_shape=(4, 2048),
                dtype=torch.bfloat16,
            ),
            # attention_mask
            torch_tensorrt.Input(
                min_shape=(2, 512),
                opt_shape=(4, 2048),
                max_shape=(8, 2048),
                # attention_mask is long in HF; TRT doesn't have int64 tensors,
                # Torch-TensorRT will insert casts if needed. Leaving as int64 is fine,
                # but int32 tends to be cleaner. Pick ONE consistently with your export.
                dtype=torch.int32,  # or torch.int32 if you cast before export
            ),
        ],
        enabled_precisions={torch.bfloat16},
        require_full_compilation=True,
        assume_dynamic_shape_support=True,
        optimization_level=5,
        min_block_size=1,
        workspace_size=8 << 30,
        use_fast_partitioner=True,
        enable_experimental_decompositions=True,
        cache_built_engines=True,
        reuse_cached_engines=True,
        max_aux_streams=8,
        num_avg_timing_iters=5,
        hardware_compatible=True,
        sparse_weights=True,
        debug=False
    )

    torch.save(trt_model, "modernbert_trt_aot.pt")
    print("✅ Saved AOT TensorRT model: modernbert_trt_aot.pt")

    return trt_model

if __name__ == "__main__":
    compile_with_aot_torch_tensorrt_dynamo("/home/ubuntu/bf16_checkpoints")
