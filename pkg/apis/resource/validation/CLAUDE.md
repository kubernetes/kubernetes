# Package: validation

## Purpose
Provides comprehensive validation logic for all DRA (Dynamic Resource Allocation) types including ResourceClaim, DeviceClass, ResourceSlice, and DeviceTaintRule.

## Key Variables
- `ResourceNormalizationRules`: Regex rules to map v1beta1 flattened paths to v1 nested paths

## Key Functions

### ResourceClaim Validation
- `ValidateResourceClaim`: Full validation for ResourceClaim objects
- `ValidateResourceClaimUpdate`: Validates updates (spec is immutable)
- `ValidateResourceClaimStatusUpdate`: Validates status updates (allocation, reservations)

### DeviceClass Validation
- `ValidateDeviceClass`: Validates DeviceClass objects
- `ValidateDeviceClassUpdate`: Validates updates (immutability checks)

### ResourceSlice Validation
- `ValidateResourceSlice`: Validates driver-published resource information
- `ValidateResourceSliceUpdate`: Validates updates

### DeviceTaintRule Validation
- `ValidateDeviceTaintRule`: Validates device taint rules
- `ValidateDeviceTaintRuleUpdate`: Validates updates

### Request Validation
- `validateDeviceRequest`: Validates device requests with selectors, allocation mode, tolerations
- `validateDeviceConstraint`: Validates constraints (MatchAttribute, DistinctAttribute)
- `validateDeviceSelector`: Validates CEL selectors with cost limits

### Device Validation
- `validateDevice`: Validates device definitions with attributes, capacities, taints
- `validateDeviceAttribute`: Validates attribute types (int, bool, string, version)
- `validateCELDeviceSelector`: Compiles and validates CEL expressions

## Key Constants
- Max sizes: 32 requests, 32 constraints, 32 selectors, 128 devices per slice
- CEL expression max length: 10Ki, max cost: 1000000

## Notes
- Uses CEL environment from `dynamic-resource-allocation/cel`
- Feature gates control which fields are validated (DRADeviceTaints, DRAAdminAccess, etc.)
- Extensive uniqueness and reference validation for request names
