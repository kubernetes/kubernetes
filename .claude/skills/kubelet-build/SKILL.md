---
name: kubelet-build
description: Build the kubelet binary. Use when compiling kubelet, creating debug builds, or cross-compiling for different platforms.
allowed-tools:
  - Bash
  - Read
---

# Kubelet Build

Build the Kubernetes kubelet binary.

## Instructions

When the user wants to build kubelet:

1. **Standard build**:
```bash
make kubelet
```

2. **Report the output location**:
   - Binary: `_output/bin/<os>/<arch>/kubelet`

## Build Commands

### Standard Build
```bash
# Build kubelet for current platform
make kubelet

# Verbose output
make kubelet GOFLAGS=-v
```

### Debug Build
```bash
# Keep debug symbols (for debugging with delve/gdb)
make kubelet DBG=1
```

### Cross-Platform Build
```bash
# Build for specific platform
make all WHAT=cmd/kubelet KUBE_BUILD_PLATFORMS=linux/arm64

# Build for all platforms
make cross WHAT=cmd/kubelet
```

### Build in Docker Container
```bash
# Hermetic build in container
build/run.sh make kubelet

# Cross-compile in container
build/run.sh make all WHAT=cmd/kubelet KUBE_BUILD_PLATFORMS=linux/amd64
```

## Build Output

| Build Type | Location |
|------------|----------|
| Standard | `_output/bin/<os>/<arch>/kubelet` |
| Local | `_output/local/bin/kubelet` |
| Release | `_output/release-tars/` |

## Build Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `GOFLAGS` | Go compiler flags | `-v` for verbose |
| `DBG` | Debug build | `1` to keep symbols |
| `KUBE_BUILD_PLATFORMS` | Target platforms | `linux/amd64,linux/arm64` |
| `WHAT` | Specific target | `cmd/kubelet` |

## Examples

User: "Build kubelet"
```bash
make kubelet
```

User: "Build kubelet for ARM64"
```bash
make all WHAT=cmd/kubelet KUBE_BUILD_PLATFORMS=linux/arm64
```

User: "Build kubelet with debug symbols"
```bash
make kubelet DBG=1
```

## Verifying the Build

```bash
# Check the binary exists
ls -la _output/bin/*/amd64/kubelet

# Check version
_output/bin/linux/amd64/kubelet --version
```

## Clean Build

```bash
# Remove all build artifacts
make clean

# Or specifically
rm -rf _output/
```
