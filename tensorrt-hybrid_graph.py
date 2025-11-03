import os

# --- set env BEFORE importing torch ---
os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
# must be CSV string, not list
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "cublas,triton"  # or "cublas"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- torch configs ---
torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.triton.cudagraphs = False
# must be STRING, not list
torch._inductor.config.max_autotune_gemm_backends = "cublas,triton"        # or "cublas"
torch._inductor.config.max_autotune_pointwise = False
torch._inductor.config.coordinate_descent_tuning = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# --- model ---
ckpt = "/home/ubuntu/bf16_checkpoints"
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModelForSequenceClassification.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()

# --- helpers ---
def mark_dyn(*tensors):
    for t in tensors:
        torch._dynamo.maybe_mark_dynamic(t, 0)
        if t.ndim > 1:
            torch._dynamo.maybe_mark_dynamic(t, 1)

# --- sample inputs ---
B0, S0 = 2, 2048
vocab = model.config.vocab_size
ids = torch.randint(0, vocab, (B0, S0), device="cuda", dtype=torch.long)
# use integer mask (HF convention) to avoid extra casts
msk = torch.ones((B0, S0), device="cuda", dtype=torch.long)
mark_dyn(ids, msk)

# --- compile ---
compiled = torch.compile(
    model,
    backend="inductor",
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)

# warmup compile
_ = compiled(input_ids=ids, attention_mask=msk)

# --- mixed shape runs ---
for b, s in [(2, 4096), (4, 2048), (4, 4096)]:
    ids = torch.randint(0, vocab, (b, s), device="cuda", dtype=torch.long)
    msk = torch.ones((b, s), device="cuda", dtype=torch.long)
    mark_dyn(ids, msk)
    _ = compiled(input_ids=ids, attention_mask=msk)
