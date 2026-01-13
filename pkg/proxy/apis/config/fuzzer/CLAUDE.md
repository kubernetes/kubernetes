# Package: fuzzer

## Purpose
The `fuzzer` package provides fuzzing functions for kube-proxy configuration types, used in API round-trip testing.

## Key Functions

- **Funcs**: Returns fuzzer functions for kube-proxy configuration types.
- **generateRandomIP**: Generates random IPv4 or IPv6 addresses.
- **generateRandomCIDR**: Generates random CIDR ranges.
- **getRandomDualStackCIDR**: Generates random dual-stack CIDR pairs.

## Behavior

The fuzzer populates KubeProxyConfiguration with random but valid values:
- Random bind addresses and ports.
- Random CIDR ranges for cluster configuration.
- Random conntrack settings (MaxPerCore, Min, timeouts).
- Random feature gate settings.
- Random masquerade bit values.

## Design Notes

- Used by Kubernetes API machinery for round-trip testing.
- Ensures configuration serialization/deserialization works correctly.
- Uses randfill library for random data generation.
- Generates valid network addresses and CIDRs.
