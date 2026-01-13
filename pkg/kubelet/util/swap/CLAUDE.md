# Package: swap

## Purpose
The `swap` package provides utilities for detecting swap status and tmpfs noswap option support on Linux systems.

## Key Functions

- **IsSwapOn**: Detects whether swap is enabled by inspecting `/proc/swaps`. Returns false on Windows. Caches result (only checks once).
- **IsTmpfsNoswapOptionSupported**: Checks if the `noswap` mount option is supported for tmpfs. Caches result (only checks once).

## Constants

- **TmpfsNoswapOption**: `"noswap"` - the mount option string for disabling swap on tmpfs.

## Detection Logic

### IsSwapOn
1. Returns false immediately on Windows.
2. Reads `/proc/swaps` file.
3. If file doesn't exist, assumes swap is disabled.
4. If more than one line (beyond headers), swap is enabled.

### IsTmpfsNoswapOptionSupported
1. Returns false on Windows.
2. Returns false if running in a user namespace (not supported).
3. If kernel >= 6.3, returns true (known support).
4. Otherwise, attempts test mount with noswap option.

## Design Notes

- Results are cached via `sync.Once` to avoid repeated system calls.
- Kernel version 6.3+ introduced tmpfs noswap support.
- User namespaces don't support tmpfs noswap (Linux limitation).
- Used by kubelet to configure memory-backed volumes appropriately.
