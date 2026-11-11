#!/usr/bin/env bash
# Run with sudo. Keeps the existing Ollama installation and model files.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

[[ $EUID -eq 0 ]] || { echo 'Run this script with sudo in your terminal.'; exit 1; }
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
archive="$work/downloads/ollama-linux-amd64-v0.40.16.tar.zst"
prefix=/opt/ollama-0.40.16
override=/etc/systemd/system/ollama.service.d/90-qwen-local.conf
checksum=87bfafb2acbc89b2e6a1be0a9043de6aa75fb450696f86bc426e65fc448f788d

[[ -x /usr/bin/zstd ]] || { echo 'The existing system zstd executable is missing.'; exit 1; }
[[ ! -e "$prefix" && ! -e "$override" ]] || {
  echo 'The new installation path or override already exists. Stop for inspection before rerunning.'
  exit 1
}
[[ $(df --output=avail -B1 / | tail -n1) -ge 12884901888 ]] || {
  echo 'Less than 12 GiB is available. Stop for a fresh space check.'; exit 1;
}
echo 'Secure Boot state:'
mokutil --sb-state

# Use the existing official Ubuntu sources for this invocation only.
# Other repository and preference files are left untouched.
scratch=$(mktemp -d /tmp/qwen-ubuntu-install.XXXXXX)
mkdir "$scratch/empty"
printf 'Package: *\nPin: release o=Ubuntu\nPin-Priority: 990\n' > "$scratch/ubuntu.pref"
apt_options=(
  -o Dir::Etc::sourcelist=/etc/apt/sources.list
  -o "Dir::Etc::sourceparts=$scratch/empty"
  -o "Dir::Etc::preferences=$scratch/ubuntu.pref"
  -o "Dir::Etc::preferencesparts=$scratch/empty"
)
packages=(
  linux-generic-hwe-22.04=6.8.0-138.138~22.04.1
  linux-modules-nvidia-595-generic-hwe-22.04=6.8.0-138.138~22.04.1+3
  nvidia-driver-595=595.99.02-0ubuntu0.22.04.1
)
apt-get "${apt_options[@]}" update
apt-get "${apt_options[@]}" --simulate --no-remove --no-install-recommends \
  install "${packages[@]}" > "$scratch/package-plan.txt"
cat "$scratch/package-plan.txt"
/usr/bin/python3 - "$scratch/package-plan.txt" <<'PY'
import pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
for line in lines:
    if line.startswith('Remv '):
        raise SystemExit('Stopped: the package plan would remove a package.')
    if line.startswith('Inst ') and ('dkms' in line.split()[1] or 'Ubuntu:' not in line):
        raise SystemExit('Stopped: unexpected DKMS or non-Ubuntu package: ' + line)
PY
apt-get "${apt_options[@]}" --yes --no-remove --no-install-recommends \
  install "${packages[@]}"

echo 'Checking the new kernel module signing certificate...'
modinfo -k 6.8.0-138-generic -F version nvidia
signer=$(modinfo -k 6.8.0-138-generic -F signer nvidia)
printf 'NVIDIA module signer: %s\n' "$signer"
[[ -n "$signer" ]] || { echo 'The NVIDIA module has no signer. Stop for inspection.'; exit 1; }

echo 'Checking the staged official Ollama release...'
# The agent may still be downloading the approved archive while APT runs.
for attempt in {1..360}; do
  if [[ -f "$archive" && $(stat -c %s "$archive") -eq 1545767640 ]]; then
    break
  fi
  if (( attempt % 6 == 1 )); then
    echo 'Waiting for the approved Ollama archive download to finish...'
  fi
  sleep 5
done
printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --strict

echo 'Installing the official Ollama archive in its own directory...'
install -d -m 755 "$prefix"
tar --use-compress-program=/usr/bin/zstd --extract --file="$archive" \
  --directory="$prefix" --no-same-owner
"$prefix/bin/ollama" --version

# Only the service uses the new binary. The old binary and libraries remain.
install -d -m 755 /etc/systemd/system/ollama.service.d
cat > "$override" <<'UNIT'
[Service]
ExecStart=
ExecStart=/opt/ollama-0.40.16/bin/ollama serve
Environment="OLLAMA_CONTEXT_LENGTH=4096"
Environment="OLLAMA_NUM_PARALLEL=1"
UNIT
chmod 644 "$override"
systemctl daemon-reload
systemctl restart ollama
for attempt in {1..20}; do
  if curl --fail --silent http://127.0.0.1:11434/api/version; then
    printf '\n'
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:11434/api/version
printf '\n'
df -h /
echo 'System installation finished. The model has not been downloaded.'
echo 'Save your work, then run: sudo reboot'
echo 'After reboot, return to the conversation so we can verify the GPU and test the exact model.'
