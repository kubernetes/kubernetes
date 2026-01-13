# Package: validation

## Purpose
Provides validation logic for RuntimeClass objects in the node.k8s.io API group.

## Key Variables
- `NodeNormalizationRules`: Regex rules to normalize field paths between API versions (maps `spec.runtimeHandler` to `handler`)

## Key Functions

### RuntimeClass Validation
- `ValidateRuntimeClass(rc *node.RuntimeClass)`: Validates a RuntimeClass object
  - Validates object metadata (DNS subdomain name)
  - Requires non-empty handler field (must be DNS label)
  - Validates optional Overhead and Scheduling fields

- `ValidateRuntimeClassUpdate(new, old *node.RuntimeClass)`: Validates updates
  - Validates metadata updates
  - Ensures handler field is immutable

### Helper Validation Functions
- `validateOverhead`: Validates Overhead by reusing ResourceRequirements validation
- `validateScheduling`: Validates Scheduling nodeSelector labels and tolerations
- `validateTolerations`: Validates tolerations and ensures uniqueness (no duplicates)

## Validation Rules
- Handler must be a valid DNS label (RFC 1123)
- Handler is immutable after creation
- Tolerations must be unique (by key, operator, value, effect)
- NodeSelector must have valid label key/value pairs

## Notes
- Uses declarative validation markers for some fields
- Reuses core validation logic for resource requirements and tolerations
