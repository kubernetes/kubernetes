# Package: validation

## Purpose
Validation functions for v1 (external) core API types, used to validate user-provided resource specifications.

## Key Functions

### Resource Validation
- **ValidateResourceRequirements(requirements, fldPath)**: Validates resource limits/requests including:
  - Valid resource names
  - Non-negative quantities
  - Requests <= limits (equality required for non-overcommittable resources like GPUs)

- **ValidateContainerResourceName(value, fldPath)**: Validates container resource names are standard or valid extended resources.

- **ValidateResourceQuantityValue(resource, value, fldPath)**: Validates quantity is non-negative and integer for integer resources.

- **ValidateNonnegativeQuantity(value, fldPath)**: Ensures quantity >= 0.

### Log Options
- **ValidatePodLogOptions(opts)**: Validates PodLogOptions:
  - TailLines >= 0
  - LimitBytes >= 1
  - SinceSeconds >= 1
  - Only one of SinceSeconds or SinceTime
  - Stream must be "Stdout", "Stderr", or "All"

### Port Validation
- **AccumulateUniqueHostPorts(containers, accumulator, fldPath)**: Checks for duplicate host ports across containers.

## Design Notes

- Uses field.ErrorList for accumulating validation errors.
- Works with v1 external types from k8s.io/api/core/v1.
- Delegates to internal validation for some checks.
- Extended resources must follow qualified name rules.
