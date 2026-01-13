# Package: v1

## Purpose
Provides the v1 (stable) versioned API registration and custom conversions for the policy API group.

## Key Constants
- `GroupName`: "policy"
- `SchemeGroupVersion`: policy/v1

## Key Functions

### Conversion Functions
- `Convert_v1_PodDisruptionBudget_To_policy_PodDisruptionBudget`: Handles selector conversion, stripping v1beta1 compatibility labels unless it's a special match-all/match-none selector

### Registration
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme
- `RegisterDefaults`: Registers defaulting functions

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal policy types
- `+k8s:defaulter-gen`: Generates defaulting functions

## Notes
- External types sourced from `k8s.io/api/policy/v1`
- In v1, empty selector {} matches all pods (different from v1beta1)
- Conversion logic preserves special selector semantics between versions
