"""On-demand Linux and NVIDIA statistics, using only the standard library."""
from pathlib import Path
import subprocess


def system_memory():
    try:
        values = {line.split(":")[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines()}
        return {"total": values["MemTotal"], "available": values["MemAvailable"],
                "used": values["MemTotal"] - values["MemAvailable"],
                "swap_used": values["SwapTotal"] - values["SwapFree"], "swap_total": values["SwapTotal"]}
    except (OSError, ValueError, KeyError):
        return None


def gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True)
        gpus = []
        for line in result.stdout.splitlines():
            name, used, total, utilization = [part.strip() for part in line.split(",")]
            gpus.append({"name": name, "used_MiB": float(used), "total_MiB": float(total),
                         "utilization": utilization})
        return gpus
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def format_stats(client, last_metrics=None):
    lines = []
    memory = system_memory()
    if memory:
        gib = lambda key: memory[key] / 2**30
        lines.append(f"RAM: {gib('used'):.2f} GiB used / {gib('total'):.2f} GiB total; {gib('available'):.2f} GiB available (includes reclaimable cache)")
        lines.append(f"Swap: {gib('swap_used'):.3f} / {gib('swap_total'):.2f} GiB")
    else:
        lines.append("RAM/swap: unavailable")
    gpus = gpu_stats()
    for gpu in gpus:
        lines.append(f"GPU {gpu['name']}: VRAM {gpu['used_MiB']:.0f}/{gpu['total_MiB']:.0f} MiB; utilization {gpu['utilization']}%")
    if not gpus:
        lines.append("GPU metrics: unavailable (nvidia-smi missing or failed)")
    try:
        models = client.get("/api/ps", timeout=3).get("models", [])
        model = next((item for item in models if item.get("name") == client.model or item.get("model") == client.model), None)
        if model and model.get("size", 0) > 0 and "size_vram" in model:
            gpu_percent = max(0, min(100, round(100 * model["size_vram"] / model["size"])))
            lines.append(f"Ollama placement: {100-gpu_percent}% CPU / {gpu_percent}% GPU (memory allocation estimate, not utilization); loaded context {model.get('context_length', 'unknown')}")
        elif not model:
            lines.append("Ollama placement: model not currently loaded")
        else:
            lines.append("Ollama placement: allocation metrics unavailable")
    except (OSError, ValueError, RuntimeError):
        lines.append("Ollama placement: unavailable")
    metrics = last_metrics or {}
    if metrics:
        wall = metrics.get("wall_seconds")
        count = metrics.get("eval_count")
        duration = metrics.get("eval_duration", 0)
        lines.append(f"Last response: {wall:.2f} s total" if isinstance(wall, (int, float)) else "Last response time: unavailable")
        lines.append(f"Generated tokens (thinking + answer): {count}" if count is not None else "Last token count: unavailable")
        lines.append(f"Generation: {count / duration * 1e9:.2f} tokens/s" if count is not None and duration else "Last tokens/s: unavailable")
        if metrics.get("done_reason"):
            lines.append(f"Last stop reason: {metrics['done_reason']}")
    else:
        lines.append("Last response: no saved metrics in this conversation")
    return "\n".join(lines)
