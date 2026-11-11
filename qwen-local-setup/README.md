# Qwen local setup status

Approved: Ubuntu NVIDIA 595 driver with signed HWE kernel modules, kernel
6.8.0-138, Ollama 0.40.16, and qwen3.5:9b-q4_K_M.

Installed successfully through Ubuntu's graphical administrator authentication:

- NVIDIA driver 595.99.02 from Ubuntu.
- Kernel 6.8.0-138 and matching NVIDIA modules signed by
  `Canonical Ltd. Kernel Module Signing`.
- Official Ollama 0.40.16 archive; published SHA-256 checksum verified.
- Existing Ollama service is active with the new binary, 4096 context, and one
  parallel request. `/api/version` confirms 0.40.16.
- Existing `deepseek-r1:1.5b` remains present.
- Secure Boot remains enabled. No MOK enrollment was requested.
- Free root filesystem space after setup: 16,768,876,544 bytes (15.6 GiB).

Reboot completed. Kernel 6.8.0-138 is running and Secure Boot is enabled.
`nvidia-smi` confirms an NVIDIA GeForce RTX 2050 with 4096 MiB VRAM and
driver 595.99.02. Approximately 326 MiB VRAM is reserved by the driver;
Ollama detects CUDA compute capability 8.6 and about 3.6 GiB available VRAM.
Before downloading the model, root filesystem free space was
16,859,152,384 bytes (15.7 GiB), and RAM available was about 11 GiB.

**Setup and testing are complete.** See [RESULTS.md](RESULTS.md) for measured
results and commands. The exact model is installed, reports Q4_K_M, and passed
text-only inference at 4096 context with thinking disabled. Warm generation
speed was 7.73 tokens/second. Ollama reports 67% CPU / 33% GPU; logs confirm
7 of 33 layers offloaded. Peak process RAM was 4.92 GiB and total GPU memory
used peaked at 2979 MiB. No more installation or testing is pending.
Do not rerun the installer.

The script uses Ubuntu APT repositories only for this invocation. It installs
the approved signed kernel/driver combination with required dependencies,
without package removal or optional recommended packages. Other repositories
are not edited. Existing kernels, files, models, and the old Ollama installation
are retained. The new official release lives in `/opt/ollama-0.40.16`; a new
systemd drop-in points the existing Ollama service at that binary and sets a
4096-token context and one parallel request. The old `ollama` command in PATH
is retained; use `/opt/ollama-0.40.16/bin/ollama` for the new client.

Secure Boot stays enabled. Canonical-signed precompiled modules should not
require MOK enrollment. If an enrollment prompt unexpectedly appears, report
it; do not disable Secure Boot.

Raw responses and resource samples are saved in `test-results.json`.
Model metadata is saved in `model-details.json`. Start with the tested 4096
context; 8192 context and long prompts were not tested. Final disk free space
was 17,117,949,952 bytes (15.94 GiB). The original DeepSeek model remains.

Official sources:

- https://ubuntu.com/server/docs/nvidia-drivers-installation/
- https://docs.ollama.com/linux
- https://github.com/ollama/ollama/releases/tag/v0.40.16
- https://ollama.com/library/qwen3.5:9b-q4_K_M
