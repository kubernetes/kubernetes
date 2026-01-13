# Package: v1

## Purpose
Provides the v1 (stable) versioned API registration, defaults, and system priority class definitions for the scheduling.k8s.io API group.

## Key Constants
- `GroupName`: "scheduling.k8s.io"
- `SchemeGroupVersion`: scheduling.k8s.io/v1

## System Priority Classes
Auto-created at cluster bootstrap:
- `system-node-critical`: Value 2,000,001,000 - Used for node-critical pods
- `system-cluster-critical`: Value 2,000,000,000 - Used for cluster-critical pods

## Key Functions

### Defaulting Functions
- `SetDefaults_PriorityClass`: Defaults PreemptionPolicy to PreemptLowerPriority if nil

### Helper Functions
- `SystemPriorityClasses()`: Returns deep copies of system priority classes
- `SystemPriorityClassNames()`: Returns names of system priority classes
- `IsKnownSystemPriorityClass(name, value, globalDefault)`: Validates system priority class

### Registration
- `Resource(resource string)`: Returns qualified GroupResource
- `AddToScheme`: Registers types with a scheme

## Code Generation Markers
- `+k8s:conversion-gen`: Generates conversion to/from internal scheduling types
- `+k8s:defaulter-gen`: Generates defaulting functions

## Notes
- External types sourced from `k8s.io/api/scheduling/v1`
- GA (stable) version of PriorityClass
- Validation ensures system priority class consistency
