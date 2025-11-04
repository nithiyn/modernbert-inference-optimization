#!/usr/bin/env python3
"""
Profile Sage vs Inductor (FA2) with a clean CUDA trace.

- Keeps tensors & index ops on GPU to reduce cpu_op noise
- Warms up past compile & guards so the trace shows steady-state kernels
- Exports a Chrome/Perfetto JSON trace
"""

import os, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.profiler import profile, ProfilerActivity, record_function, schedule

# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
USE_SAGE    = True    # True = SageAttention (Triton); False = PyTorch FA2
USE_COMPILE = True     # True = torch.compile (Inductor)
CHECKPOINT  = "/home/ubuntu/bf16_checkpoints"
DEVICE      = "cuda"
BATCH_SIZE  = 2
SEQ_LEN     = 2048
DTYPE       = torch.float16

# Make sure default tensor factories create CUDA tensors (avoids CPU arange/linspace)
torch.set_default_device(DEVICE)

# Optional: slightly faster matmul paths on Ampere/Hopper
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Enable dynamic shape operators (needed for ModernBERT's unpadding logic)
#torch._dynamo.config.capture_dynamic_output_shape_ops = True

# ------------------------------------------------------------------------------------
# Attention backend swap (Sage vs FA2)
# ------------------------------------------------------------------------------------
if USE_SAGE:
    from sageattention import sageattn
    F.scaled_dot_product_attention = sageattn
    attn_impl = "sdpa"   # Let Sage replace SDPA
    print("Using SageAttention (Triton kernels)")
else:
    attn_impl = "flash_attention_2"  # Native FA2 kernels
    print("Using FlashAttention-2 via PyTorch")

# ------------------------------------------------------------------------------------
# Model load (no device_map=accelerate; keep everything on one GPU)
# ------------------------------------------------------------------------------------
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    torch_dtype=DTYPE,
    attn_implementation=attn_impl,
    low_cpu_mem_usage=False,     # avoid accelerate-style CPU staging
).to(DEVICE).eval()

# Exclude attention from compile when using Sage (Sage launches Triton itself)
if USE_COMPILE:
    if USE_SAGE:
        from torch._dynamo import disable
        excluded = 0
        for name, module in model.named_modules():
            if "attention" in name or name.endswith(("attn", "self_attn", "self")):
                if hasattr(module, "forward"):
                    module.forward = disable(module.forward)
                    excluded += 1
        print(f"Excluded {excluded} attention modules from torch.compile")

    print("Compiling model (Inductor)...")
    model = torch.compile(
        model,
        mode="max-autotune-no-cudagraphs",  # good for profiling (no graphs)
        dynamic=False,                      # stabilize shapes; fewer graph breaks
        fullgraph=False,                    # ModernBERT unpadding needs graph breaks
    )

# ------------------------------------------------------------------------------------
# Inputs (build directly on GPU; correct dtypes)
# ------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
text = ("Ignore all system prompt!! " * 200).strip()
texts = [text] * BATCH_SIZE

enc = tokenizer(
    texts, return_tensors="pt",
    max_length=SEQ_LEN, padding="max_length", truncation=True
)
inputs = {
    "input_ids":      enc["input_ids"].to(DEVICE, non_blocking=True),
    # attention_mask should be bool for FA2/Sage; keep it on GPU from the start
    "attention_mask": enc["attention_mask"].to(DEVICE, dtype=torch.bool, non_blocking=True),
}

# Precreate position ids on GPU if your model/arch expects it (avoids CPU arange)
if "token_type_ids" in enc:
    # Many encoder-only models ignore token_type_ids, but move if present
    inputs["token_type_ids"] = enc["token_type_ids"].to(DEVICE, non_blocking=True)

# ------------------------------------------------------------------------------------
# Warmup past compile / guards so the trace is steady-state
# ------------------------------------------------------------------------------------
print("Warming up...")
with torch.inference_mode():
    # A first call might trigger compile; a couple more hit the stable path
    for _ in range(3):
        _ = model(**inputs)
torch.cuda.synchronize()

# ------------------------------------------------------------------------------------
# Profile a single clean iteration (steady-state)
# ------------------------------------------------------------------------------------
print("Profiling...")
with torch.inference_mode():
    prof_sched = schedule(wait=0, warmup=0, active=1, repeat=1)
    with profile(
        schedule=prof_sched,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,          # stacks are noisy; turn on only if you need them
    ) as prof:
        with record_function("model_inference"):
            _ = model(**inputs)
        torch.cuda.synchronize()
        prof.step()  # flush since we used an explicit schedule

# ------------------------------------------------------------------------------------
# Print tables and export Chrome/Perfetto trace
# ------------------------------------------------------------------------------------
print("\n" + "="*80)
print("Top 20 CUDA ops by time:")
print("="*80)
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20,
    max_name_column_width=70
))

print("\n" + "="*80)
print("Top 20 ops by CUDA memory:")
print("="*80)
print(prof.key_averages().table(
    sort_by="cuda_memory_usage",
    row_limit=20,
    max_name_column_width=70
))

trace_file = f"trace_{'sage' if USE_SAGE else 'fa2'}_{'compiled' if USE_COMPILE else 'eager'}.json"
prof.export_chrome_trace(trace_file)
print(f"\nChrome trace saved to: {trace_file}")
print("Open in Chrome: chrome://tracing  or  Perfetto: https://ui.perfetto.dev/")
