# Gemma 4 12B on Tesla P100 — MTP speculative decoding patch + assistant converter

Two artifacts for running **Gemma 4 12B (QAT Q4_0)** on a **Tesla P100 (sm_60)** at usable speed with
**MTP (multi-token prediction) speculative decoding**:

| File | What it is |
|---|---|
| `llama-p100-compat.patch` | Patch for upstream llama.cpp: a GP100/pascal MMVQ parameter table that raises decode throughput ~25% |
| `convert_assistant.py` | Standalone (stdlib-only) converter: Gemma 4 12B IT assistant (MTP draft) from HF safetensors to GGUF |
| `test_convert_assistant.py` | Regression test for the converter (RoPE payload, tensor alignment) |

## Results on the reference machine

P100 PCIe 16 GB over Thunderbolt eGPU (Gen3 x4, CUDA 12.6, driver 580), Q4_0 weights, 8K context,
`temp=1.0 / top_p=0.95`, reasoning on, English chat:

| Scenario | Decode |
|---|---|
| stock llama.cpp, no speculation | 29.5 t/s |
| + patch, no speculation | 36.3 t/s |
| + patch + MTP draft (Q4_0), chat | 47 t/s |
| + patch + MTP draft (Q4_0), code | 53 t/s |

Prefill: 411 t/s (pp512), 382 t/s (real 4.6K prompt), ~269 t/s (32K), ~145 t/s (128K).
Long context: 192K + image input + MTP fits in 14.9 GB of 16.3 GB VRAM.

## Requirements

- llama.cpp at **b94959a** (or nearby) with CUDA support
- CUDA toolkit that can target `sm_60` (12.6 tested; 13.x dropped Pascal)
- Models from Google's gated HF repos:
  - `gemma-4-12b-it-qat-q4_0.gguf` (main, premade GGUF)
  - `mmproj-gemma-4-12b-it-qat-q4_0.gguf` (vision, premade)
  - assistant BF16 safetensors (source for the MTP draft)

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout b94959a
git apply ../llama-p100-compat.patch

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli llama-server llama-quantize -j8
```

What the patch does: llama.cpp's `mul_mat_vec_q` dispatch table picks rows-per-block per
architecture; the table has no Pascal row, so sm_60 falls back to a generic (slow) choice.
The patch adds `GGML_CUDA_CC_PASCAL` entries — rows/block 2 for M=1 (4 for K-quants) plus a
Pascal branch for warp count — which is a pure scheduling change: no math, no precision impact.

## Convert the MTP assistant draft

```bash
# 1) BF16 safetensors -> GGUF (stdlib only, no dependencies)
python3 convert_assistant.py <assistant_model_dir_or_safetensors> \
    models/gemma-4-12b-it-assistant-bf16.gguf

# 2) quantize the draft to Q4_0
llama-quantize models/gemma-4-12b-it-assistant-bf16.gguf \
    models/gemma-4-12b-it-assistant-q4_0.gguf Q4_0
```

The converter copies linear weights as BF16, norms as F32, `rope_freqs` raw, and tokenizer
metadata verbatim from the main model's GGUF conventions. A Q4_0 draft (323 MB) is enough —
drafts are verified by the target model, so a smaller draft only affects acceptance rate,
never output quality.

## Run

```bash
llama-server \
  -m models/gemma-4-12b-it-qat-q4_0.gguf \
  --mmproj models/mmproj-gemma-4-12b-it-qat-q4_0.gguf \
  --spec-type draft-mtp -md models/gemma-4-12b-it-assistant-q4_0.gguf \
  --spec-draft-n-max 6 --spec-draft-p-min 0.95 \
  -ngl 99 -ngld 99 -fit off -c 196608 -np 2 -kvu -b 2048 -ub 2048 \
  --host 0.0.0.0 --port 8080
```

Notes:

- `-kvu` (unified KV pool) is required when `-np` is set explicitly. Without it llama-server
  statically splits the context per slot (196K / 2 = 98K each) and, worse, two concurrent long
  prefills exhaust the compute-buffer pool and the server **aborts with a CUDA OOM**. With
  `-kvu` all slots draw from one 196K-token pool, so a single conversation can use the whole
  window and concurrent requests queue instead of overflowing.
- `-b 2048 -ub 2048` is worth ~20% prefill at 8K on this GPU (ub=1024 is *slower* than 512 —
  don't assume monotonicity). It costs ~2.2 GB of compute buffers vs ub=512, which is why
  192K + two slots only fits with a 196K (not 256K) pool.
- `--spec-draft-p-min 0.95` with `n-max 6` is the measured sweet spot; the draft stops early
  when its confidence drops, which keeps the target verification batch small.
- `-fit off` is required: llama.cpp's new auto-fit step builds the draft context without
  `ctx_other`, and draft-MTP init fails.
- VRAM on a 16 GB P100: 14.2 GB after load, ~14.4 GB peak with two concurrent requests.
  For more headroom use `-c 131072` (~11.5 GB).
- Sampling: the model expects `temp=1.0, top_p=0.95` for its default behavior; thinking-mode
  responses put the reasoning in `reasoning_content` (OpenAI-compatible API) — `max_tokens`
  must exceed the thinking length or `content` will be empty.

## License

- MIT
- Gemma weights: Google Gemma Terms of Use — this repo ships none.
