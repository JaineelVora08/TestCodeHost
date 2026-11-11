# Qwen3.5 local test results

Successfully tested `qwen3.5:9b-q4_K_M` on the HP Victus laptop.

| Item | Measured result |
|---|---|
| Model | Qwen3.5 9B, GGUF Q4_K_M |
| Model digest | `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90` |
| Ollama | 0.40.16; the current model reports a minimum of 0.30.0 |
| Kernel / driver | 6.8.0-138-generic / NVIDIA 595.99.02 |
| GPU | NVIDIA GeForce RTX 2050, 4096 MiB VRAM |
| Secure Boot | Enabled; Canonical-signed NVIDIA modules |
| Context / concurrency | 4096 tokens / one request |
| Thinking | Disabled for the test |
| Correctness check | 2+2 returned `4` |
| Warm generation speed | 7.73 tokens/second |
| Warm response | 180 generated tokens in 23.29 seconds of generation; 24.58 seconds total |
| First cold request | 70.58 seconds total, including 40.55 seconds loading and 29.89 seconds prompt processing |
| CPU / GPU offloading | Ollama reports 67% CPU / 33% GPU; logs confirm 7 of 33 layers offloaded |
| Peak Ollama RAM | 4.92 GiB resident RAM across service processes |
| Minimum system RAM available | 9.46 GiB |
| Peak GPU memory used | 2979 MiB total GPU use (about 2.91 GiB) |
| Ollama GPU process memory | 2956 MiB |
| GPU memory free after test | 793 MiB; driver also reserves approximately 326 MiB |
| Peak swap occupied | 1.0 MiB |
| Final disk space available | 15.94 GiB |

Memory was sampled approximately once per second. Resident RAM includes
memory-mapped model pages; Linux may also retain reclaimable file cache.
The offloading percentages describe Ollama's memory placement, not GPU
utilization or the fraction of work accelerated. The bundled vision projector
is loaded by this exact model even though the requests contained only text.
The GPU memory figure from nvidia-smi includes allocations beyond Ollama's
/api/ps size_vram field, so they should not be treated as identical measures.

Use 4096 context initially. It passed this short text test. There is enough
observed RAM headroom to consider 8192 later, but 8192 and long prompts were
not tested. Keep 4096 as the default for now. These short-response timings
are approximate and depend on prompt length and laptop load.

## Run again

The Ollama service starts automatically. The original 0.9.6 client remains
in PATH to preserve the old installation. Use the updated client's path:

```bash
/opt/ollama-0.40.16/bin/ollama run qwen3.5:9b-q4_K_M --think=false
/opt/ollama-0.40.16/bin/ollama ps
nvidia-smi
```

The service sets the 4096-token default. To release the model's memory:

```bash
/opt/ollama-0.40.16/bin/ollama stop qwen3.5:9b-q4_K_M
```

The tests requested a ten-minute keep-alive; the model unloads automatically
after that idle interval. The original DeepSeek model was retained. No UI,
chat application, extra Python packages, partition changes, or file cleanup
were performed.

Raw test responses and samples are saved in `test-results.json`;
model metadata is saved in `model-details.json`.

Ollama's API reports generation duration in nanoseconds. Speed was calculated
as eval_count / (eval_duration / 1e9):
https://docs.ollama.com/api/chat

Official context guidance: https://docs.ollama.com/context-length
