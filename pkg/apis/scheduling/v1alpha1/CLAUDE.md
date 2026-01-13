# Package: v1alpha1

## Purpose
Provides the v1alpha1 versioned API registration and defaults for the scheduling.k8s.io API group.

## Key Constants
- `GroupName`: "scheduling.k8s.io"
- `SchemeGroupVersion`: scheduling.k8s.io/v1alpha1

## Key Functions

### Defaulting Functions
- `SetDefaults_PriorityClass`: Defaults PreemptionPolicy to PreemptLowerPriority if nil

### Registration
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal scheduling types
- `+k8s:defaulter-gen`: Generates defaulting functions

## Notes
- External types sourced from `k8s.io/api/scheduling/v1alpha1`
- Early alpha version of PriorityClass; superseded by v1beta1 and v1
- Maintains backward compatibility for upgrades
