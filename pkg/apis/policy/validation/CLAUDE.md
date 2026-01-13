# Package: validation

## Purpose
Provides validation logic for PodDisruptionBudget objects and related sysctl patterns.

## Key Types

### PodDisruptionBudgetValidationOptions
- `AllowInvalidLabelValueInSelector`: Backward compatibility flag for invalid label values

## Key Functions

### PDB Validation
- `ValidatePodDisruptionBudget`: Top-level validation for PodDisruptionBudget objects
- `ValidatePodDisruptionBudgetSpec`: Validates PDB spec:
  - Ensures minAvailable and maxUnavailable are mutually exclusive
  - Validates minAvailable/maxUnavailable as positive int or percentage (0-100%)
  - Validates label selector
  - Validates UnhealthyPodEvictionPolicy (IfHealthyBudget or AlwaysAllow)
- `ValidatePodDisruptionBudgetStatusUpdate`: Validates status updates:
  - Validates conditions
  - For v1 (not v1beta1): validates non-negative integers for health metrics

### Sysctl Validation
- `IsValidSysctlPattern(name string) bool`: Validates sysctl patterns against regex
- Supports patterns with slashes, wildcards, and standard sysctl naming

## Key Constants
- `SysctlContainSlashPatternFmt`: Regex pattern for sysctl validation
- `supportedUnhealthyPodEvictionPolicies`: Set of valid eviction policies

## Validation Rules
- minAvailable/maxUnavailable: Must be positive integer or 0-100%
- Only one of minAvailable/maxUnavailable can be set
- Selector must be valid label selector
- v1 status fields must be non-negative

## Notes
- v1beta1 status updates skip numeric validation for backward compatibility
- Sysctl validation supports Linux kernel parameter naming conventions
