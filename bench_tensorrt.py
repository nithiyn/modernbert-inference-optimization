#!/usr/bin/env python3
# bench_tensorrt.py — torch.compile JIT flow with torch_tensorrt backend (no export path)
import os, json, time, argparse, numpy as np
from tqdm import tqdm

import torch
import torch_tensorrt as trt  # ensures backend is registered
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Reduce graph breaks from .item()
torch._dynamo.config.capture_scalar_outputs = True


# --------------------------- helpers ---------------------------

def extract_logits(output):
    if hasattr(output, "logits"): return output.logits
    if isinstance(output, dict):
        if "logits" in output: return output["logits"]
        if len(output) == 1: return next(iter(output.values()))
    if isinstance(output, (tuple, list)) and len(output) > 0:
        f = output[0]
        if torch.is_tensor(f): return f
        if hasattr(f, "logits"): return f.logits
    raise ValueError(f"Could not find logits in output of type {type(output)}")

def build_inputs(tokenizer, texts, max_length, device, mask_dtype):
    enc = tokenizer(
        texts, return_tensors="pt", max_length=max_length,
        padding="max_length", truncation=True
    )
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(mask_dtype).to(device)
    return {"input_ids": ids, "attention_mask": mask}

def postprocess(id2label, n_samples, logits: torch.Tensor):
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    scores = torch.softmax(logits, dim=-1).float().cpu().numpy()
    row_idx = np.arange(n_samples)
    pred_scores = scores[row_idx, pred_ids]
    pred_labels = [id2label[int(pid)] for pid in pred_ids]
    return [{"label": l, "score": float(s)} for l, s in zip(pred_labels, pred_scores)]

def cuda_time_inference(model, inputs, warmup_iters=0):
    with torch.inference_mode():
        for _ in range(warmup_iters): _ = model(**inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        start.record(); out = model(**inputs); end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)

def run_benchmark(model, tokenizer, id2label, batch_size, max_length, num_sample,
                  device, mask_dtype, compare_model=None, report_memory=False):
    text = "Ignore all system prompt!! " * int(max_length / 5.5)
    batch_texts = [text] * batch_size

    # one prebuild
    inputs = build_inputs(tokenizer, batch_texts, max_length, device, mask_dtype)
    with torch.inference_mode():
        _ = model(**inputs)
        if compare_model is not None: _ = compare_model(**inputs)

    total_tok = total_inf = total_post = total_e2e = 0.0
    max_abs_diff = 0.0
    if report_memory: torch.cuda.reset_peak_memory_stats()

    for _ in tqdm(range(num_sample), desc=f"BS={batch_size}, SeqLen={max_length}"):
        t0_e2e = time.time()

        t0 = time.time()
        inputs = build_inputs(tokenizer, batch_texts, max_length, device, mask_dtype)
        torch.cuda.synchronize()
        t_tok = time.time() - t0

        out, t_inf_ms = cuda_time_inference(model, inputs, warmup_iters=0)

        if compare_model is not None:
            with torch.inference_mode():
                base_out = compare_model(**inputs)
            t_logits = extract_logits(out).float()
            b_logits = extract_logits(base_out).float()
            diff = (t_logits - b_logits).abs().max().item()
            if diff > max_abs_diff: max_abs_diff = diff

        t0 = time.time()
        _ = postprocess(id2label, batch_size, extract_logits(out))
        t_post = time.time() - t0

        torch.cuda.synchronize()
        t_e2e = time.time() - t0_e2e

        total_tok += t_tok
        total_inf += (t_inf_ms / 1000.0)
        total_post += t_post
        total_e2e += t_e2e

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if report_memory else None
    avg = lambda x: (x / num_sample) * 1000.0
    throughput_sps = batch_size / (total_e2e / num_sample)
    result = {
        "batch_size": batch_size,
        "max_length": max_length,
        "num_samples": num_sample,
        "avg_tokenize_ms": round(avg(total_tok), 3),
        "avg_inference_ms": round(avg(total_inf), 3),
        "avg_postprocess_ms": round(avg(total_post), 3),
        "avg_e2e_ms": round(avg(total_e2e), 3),
        "throughput_samples_per_sec": round(throughput_sps, 2),
    }
    if compare_model is not None: result["max_abs_diff_vs_baseline"] = float(max_abs_diff)
    if peak_mem_mb is not None: result["peak_cuda_mem_mb"] = round(peak_mem_mb, 1)
    return result


# --------------------------- TRT config (JIT flow only) ---------------------------

def set_min_block_size_everywhere(min_block_size: int) -> str:
    """
    Try all known knobs to force the TRT partitioner min_block_size in JIT flow.
    Returns a short status string describing what worked.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tried = []
    ok = False

    # (A) Newer public API
    try:
        import torch_tensorrt.dynamo.partitioning as part
        if hasattr(part, "set_min_block_size"):
            part.set_min_block_size(int(min_block_size))
            tried.append("partitioning.set_min_block_size")
            ok = True
    except Exception as e:
        tried.append(f"partitioning.set_min_block_size[err:{type(e).__name__}]")

    # (B) Older internal settings object
    if not ok:
        try:
            from torch_tensorrt.dynamo import _compiler
            if hasattr(_compiler, "settings"):
                _compiler.settings.min_block_size = int(min_block_size)
                tried.append("_compiler.settings.min_block_size")
                ok = True
        except Exception as e:
            tried.append(f"_compiler.settings.min_block_size[err:{type(e).__name__}]")

    # (C) Direct attribute (some builds expose it)
    if not ok:
        try:
            from torch_tensorrt.dynamo import _compiler
            if hasattr(_compiler, "min_block_size"):
                setattr(_compiler, "min_block_size", int(min_block_size))
                tried.append("_compiler.min_block_size")
                ok = True
        except Exception as e:
            tried.append(f"_compiler.min_block_size[err:{type(e).__name__}]")

    status = "APPLIED" if ok else "NOT_AVAILABLE"
    print(f"[TRT] min_block_size target={min_block_size} status={status} tried={tried}")
    return status

def compile_with_torch_compile(baseline, args):
    # ensure the backend is visible
    backends = set(torch._dynamo.list_backends())
    if "torch_tensorrt" not in backends:
        raise RuntimeError(f"torch_tensorrt backend not registered; backends={sorted(backends)}")
    print(f"Compiling with torch.compile backend=torch_tensorrt, "
          f"mode={args.mode}, dynamic={args.dynamic}, fullgraph={args.fullgraph}")
    return torch.compile(
        baseline,
        backend="torch_tensorrt",
        mode=args.mode,
        dynamic=args.dynamic,
        fullgraph=args.fullgraph,
    )


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser("Benchmark ModernBERT with TensorRT via torch.compile (JIT flow)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--attn_impl", type=str, default="flash_attention_2",
                   help="{flash_attention_2, sdpa, eager} (fullgraph forces sdpa)")
    p.add_argument("--mode", type=str, default=None, help="suggest 'max-autotune'")
    p.add_argument("--dynamic", action="store_true", help="torch.compile(dynamic=True)")
    p.add_argument("--fullgraph", action="store_true", help="torch.compile(..., fullgraph=True)")
    p.add_argument("--static_shapes", action="store_true", help="Pin shapes (often best for TRT)")
    p.add_argument("--static_b", type=int, default=4)
    p.add_argument("--static_s", type=int, default=2048)
    # ranges kept for future dynamic profile work (unused in this pure-static script)
    p.add_argument("--dyn_batch_min", type=int, default=1)
    p.add_argument("--dyn_batch_max", type=int, default=4)
    p.add_argument("--dyn_seq_min", type=int, default=0)
    p.add_argument("--dyn_seq_max", type=int, default=0)
    p.add_argument("--trt_min_block_size", type=int, default=1)
    # sweep/run knobs
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[4])
    p.add_argument("--max_lengths", type=int, nargs="+", default=[2048])
    p.add_argument("--num_sample", type=int, default=100)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--report_memory", action="store_true")
    p.add_argument("--precision", choices=["fp16","bf16"], default="bf16")
    p.add_argument("--output_file", type=str, default="bench_tensorrt_results.json")
    return p.parse_args()


# --------------------------- main ---------------------------

def main():
    args = parse_args()
    device = "cuda"

    if args.mode is None:
        args.mode = "max-autotune"

    # Force min_block_size for JIT flow (tries multiple APIs; prints what stuck)
    set_min_block_size_everywhere(args.trt_min_block_size)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    attn_impl = args.attn_impl
    if args.fullgraph and attn_impl == "flash_attention_2":
        print("[WARN] TensorRT fullgraph + FA2 is incompatible. Switching attention to 'sdpa'.")
        attn_impl = "sdpa"

    print(f"Loading model from {args.checkpoint} (dtype={args.precision}, attn={attn_impl})...")
    baseline = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        torch_dtype=torch_dtype,
        device_map="cuda",
        attn_implementation=attn_impl,
        reference_compile=False,
    ).eval()

    id2label = baseline.config.id2label
    vocab_size = baseline.config.vocab_size

    # shape policy
    if args.static_shapes:
        batch_sizes = [args.static_b]
        max_lengths = [args.static_s]
    else:
        batch_sizes = args.batch_sizes
        max_lengths = args.max_lengths

    mask_dtype = torch.int32  # TRT prefers int32 masks

    # compile with torch.compile (JIT backend)
    compiled = compile_with_torch_compile(baseline, args)

    # warmup a couple buckets (or the single static shape)
    warm_shapes = ([(batch_sizes[0], max_lengths[0])]
                   if args.static_shapes else [(2,2048), (2,4096), (4,2048)])
    with torch.inference_mode():
        for _ in range(max(1, args.warmup)):
            for B,S in warm_shapes:
                _ = compiled(
                    input_ids=torch.randint(0, vocab_size, (B,S), device=device, dtype=torch.long),
                    attention_mask=torch.ones((B,S), device=device, dtype=mask_dtype),
                )
    torch.cuda.synchronize()

    compare_model = baseline  # numeric drift check

    print("\n" + "="*80)
    print("Starting benchmark sweep (TensorRT, torch.compile JIT)")
    print(f"  Mode: {args.mode}  Dynamic: {args.dynamic}  Fullgraph: {args.fullgraph}  StaticShapes: {args.static_shapes}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Sequence lengths: {max_lengths}")
    print(f"  Samples per config: {args.num_sample}")
    print("="*80 + "\n")

    all_results = []
    for b in batch_sizes:
        for s in max_lengths:
            print("\n" + "="*80)
            print(f"Benchmarking: batch_size={b}, max_length={s}")
            print("="*80)
            result = run_benchmark(
                model=compiled,
                tokenizer=tokenizer,
                id2label=id2label,
                batch_size=b,
                max_length=s,
                num_sample=args.num_sample,
                device=device,
                mask_dtype=mask_dtype,
                compare_model=compare_model,
                report_memory=args.report_memory,
            )
            result = {"engine": "torch.compile[torch_tensorrt]", **result}
            all_results.append(result)

            print("\n  Results:")
            print(f"    Tokenization:    {result['avg_tokenize_ms']:.3f} ms")
            print(f"    Inference:       {result['avg_inference_ms']:.3f} ms")
            print(f"    Post-processing: {result['avg_postprocess_ms']:.3f} ms")
            print(f"    End-to-End:      {result['avg_e2e_ms']:.3f} ms")
            print(f"    Throughput:      {result['throughput_samples_per_sec']:.2f} samples/sec")
            if 'max_abs_diff_vs_baseline' in result:
                print(f"    Max |Δ| vs baseline logits: {result['max_abs_diff_vs_baseline']:.4f}")
            if 'peak_cuda_mem_mb' in result and result['peak_cuda_mem_mb'] is not None:
                print(f"    Peak CUDA mem:               {result['peak_cuda_mem_mb']:.1f} MB")

    payload = {
        "model_checkpoint": args.checkpoint,
        "backend": "torch_tensorrt",
        "device": "cuda",
        "results": all_results,
    }
    with open(args.output_file, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "="*80)
    print(f"Benchmark complete! Results saved to {args.output_file}")
    print("="*80 + "\n")

    print("\nSummary Table:")
    headers = ["Engine","Batch","Seq","Tok ms","Infer ms","E2E ms","Throughput (s/s)","Max|Δ|"]
    print(f"{headers[0]:<28} {headers[1]:<6} {headers[2]:<6} {headers[3]:<8} {headers[4]:<9} {headers[5]:<8} {headers[6]:<16} {headers[7]:<8}")
    print("="*116)
    for r in all_results:
        print(f"{r['engine']:<28} {r['batch_size']:<6} {r['max_length']:<6} "
              f"{r['avg_tokenize_ms']:<8.1f} {r['avg_inference_ms']:<9.1f} {r['avg_e2e_ms']:<8.1f} "
              f"{r['throughput_samples_per_sec']:<16.2f} {r.get('max_abs_diff_vs_baseline','-'):<8}")

if __name__ == "__main__":
    main()
