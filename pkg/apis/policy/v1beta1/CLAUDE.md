# Package: v1beta1

## Purpose
Provides the v1beta1 versioned API registration and custom conversions for the policy API group, with special handling for selector semantics.

## Key Constants
- `GroupName`: "policy"
- `SchemeGroupVersion`: policy/v1beta1

## Key Functions

### Conversion Functions
Handle the different empty-selector semantics between v1beta1 and v1:

- `Convert_v1beta1_PodDisruptionBudget_To_policy_PodDisruptionBudget`:
  - v1beta1 empty selector {} -> internal match-none selector (preserves v1beta1 behavior)
  - v1beta1 match-all selector -> internal empty selector {}
  - Otherwise strips v1beta1 compatibility labels

- `Convert_policy_PodDisruptionBudget_To_v1beta1_PodDisruptionBudget`:
  - Internal match-none selector -> v1beta1 empty selector {}
  - Internal empty selector {} -> v1beta1 match-all selector (with special label)

### Registration
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme
- `RegisterDefaults`: Registers defaulting functions

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal policy types
- `+k8s:defaulter-gen`: Generates defaulting functions

## Notes
- In v1beta1, empty selector {} matches NO pods (historical behavior)
- This differs from v1 where empty selector {} matches ALL pods
- Conversion functions ensure consistent behavior when objects are stored/retrieved across versions
- Uses `pdb.kubernetes.io/deprecated-v1beta1-empty-selector-match` label for compatibility
