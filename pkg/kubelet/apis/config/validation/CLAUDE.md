# Package validation

Package validation provides comprehensive validation logic for KubeletConfiguration to ensure configuration correctness before the kubelet starts.

## Key Functions

- `ValidateKubeletConfiguration`: Main validation entry point that checks all configuration fields against allowed values and constraints

## Validation Scope

The package validates:
- Port numbers (healthz, kubelet, read-only)
- Numeric ranges (OOM score, GC thresholds, burst/QPS values)
- Feature gate dependencies (validates settings require their feature gates)
- Topology manager policy and scope values
- Hairpin mode settings
- Node allocatable enforcement options
- Reserved CPU/memory configurations
- Graceful shutdown settings
- Container log settings
- Swap behavior configuration
- Tracing and logging configuration

## Platform-Specific Validation

- `validation_linux.go`: Linux-specific validation (cgroup driver, memory provider)
- `validation_windows.go`: Windows-specific validation (memory provider restrictions)
- `validation_others.go`: Stub for other platforms
- `validation_reserved_memory.go`: Validates reserved memory configuration

## Design Notes

- Validates feature gate enablement before allowing dependent configuration
- Creates a local copy of feature gates merged with config-specified gates for validation
- Returns aggregated errors for all validation failures
- Cross-validates interdependent settings (e.g., image GC thresholds)
