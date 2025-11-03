# compile_aot_torch_tensorrt_dynamo.py

import torch
import torch_tensorrt
from transformers import AutoModelForSequenceClassification

# ...imports unchanged...

def compile_with_aot_torch_tensorrt_dynamo(checkpoint_path):
    print("=== AOT Compilation with torch_tensorrt.dynamo (torch.export) ===")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,   # start with FP16; switch to BF16 after success
        device_map=None,             # keep on CPU for export
        attn_implementation="eager",
        reference_compile=False
    ).eval()

    B_MIN, B_OPT, B_MAX = 2, 4, 4
    S_STATIC = 2048

    # Example tensors (CPU is fine for torch.export)
    input_ids = torch.randint(0, 50368, (B_OPT, S_STATIC), dtype=torch.long)
    attention_mask = torch.ones((B_OPT, S_STATIC), dtype=torch.int32)

    # Only batch is dynamic; seq is static 2048
    dyn = {
        "input_ids": {0: torch.export.Dim("batch", min=B_MIN, max=B_MAX)},
        "attention_mask": {0: torch.export.Dim("batch", min=B_MIN, max=B_MAX)},
    }

    print("Exporting model...")
    exported_program = torch.export.export(
        model,
        args=(),
        kwargs={"input_ids": input_ids, "attention_mask": attention_mask},
        dynamic_shapes=dyn,
    )

    print("Compiling exported program with TensorRT (guarded mode)...")
    trt_inputs = [
        # input_ids: omit dtype so TT decides and inserts casts if needed
        torch_tensorrt.Input(
            min_shape=(B_MIN, S_STATIC),
            opt_shape=(B_OPT, S_STATIC),
            max_shape=(B_MAX, S_STATIC),
        ),
        # attention_mask: explicit int32
        torch_tensorrt.Input(
            min_shape=(B_MIN, S_STATIC),
            opt_shape=(B_OPT, S_STATIC),
            max_shape=(B_MAX, S_STATIC),
            dtype=torch.int32,
        ),
    ]

    trt_model = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=trt_inputs,
        enabled_precisions={torch.float16},  # try bfloat16 after a clean FP16 build
        require_full_compilation=False,      # first pass: allow fallback
        assume_dynamic_shape_support=True,
        optimization_level=4,
        min_block_size=5,
        workspace_size=(2 << 30),            # 2 GiB
        use_fast_partitioner=True,
        enable_experimental_decompositions=True,
        cache_built_engines=True,
        reuse_cached_engines=True,
        max_aux_streams=4,
        num_avg_timing_iters=2,
        hardware_compatible=True,
        sparse_weights=False,
        debug=False,
        # tactic_sources=...  <-- removed; not available on your build
    )

    torch.save(trt_model, "modernbert_trt_aot.pt")
    print("✅ Saved AOT TensorRT model: modernbert_trt_aot.pt")
    return trt_model

if __name__ == "__main__":
    compile_with_aot_torch_tensorrt_dynamo("/home/ubuntu/bf16_checkpoints")
