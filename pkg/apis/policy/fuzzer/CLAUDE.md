# Package: fuzzer

## Purpose
Provides fuzzer functions for the policy API group, used in fuzz testing to generate random valid API objects.

## Key Functions
- `Funcs`: Returns fuzzer functions for policy types

### PodDisruptionBudgetStatus Fuzzer
Generates random status with `DisruptionsAllowed` set to 0 or 1 (valid non-negative values), ensuring fuzzed status passes validation.

## Notes
- Uses `randfill.Continue` for controlled random generation
- Limits DisruptionsAllowed to small values (0-1) to ensure valid test data
- Part of Kubernetes' fuzz testing infrastructure for API stability
