# Package: fuzzer

## Purpose
Provides fuzz testing functions for the autoscaling API group types, generating valid random instances for testing serialization and API machinery.

## Key Variables
- `Funcs`: Returns fuzzer functions for autoscaling API types

## Fuzzer Functions Provided
- `ScaleStatus`: Generates valid label selector strings
- `HorizontalPodAutoscalerSpec`: Generates complete HPA specs with:
  - MinReplicas set to random value
  - Metrics array with Pods, Object, and Resource metric sources
  - Behavior with ScaleUp (Max policy) and ScaleDown (Min policy) rules
  - Multiple scaling policies (Pods and Percent types)
- `HorizontalPodAutoscalerStatus`: Generates status with:
  - Current metrics for Pods and Resource types
  - Random quantity values and utilization percentages

## Design Notes
- Ensures fuzzed objects have valid metric configurations
- Generates both spec and status fuzzing for complete testing
- Uses randfill library for random value generation
- Critical for testing HPA serialization roundtrips
