# Package: fuzzer

## Purpose
This package provides fuzzer functions for the kubelet configuration API types. It generates valid random values for KubeletConfiguration during fuzz testing, ensuring round-trip encoding/decoding consistency.

## Key Functions

- **Funcs()**: Returns fuzzer functions for the kubeletconfig API group

## Fuzzer Behavior

The fuzzer sets realistic default values for KubeletConfiguration fields:
- Server settings (ports, addresses, TLS)
- Authentication/authorization modes and cache TTLs
- Resource management policies (CPU, memory, topology managers)
- Eviction thresholds and garbage collection settings
- Container runtime endpoints
- QoS and cgroup settings

## Design Notes

- Provides non-empty values for fields with defaults
- Prevents defaulter from changing values during round-trip tests
- Sets specific default values like OOMScoreAdj, ContainerLogMaxSize
- Handles special cases like CredentialProvider tokenAttributes
- Used by serialization tests to verify encoding stability
