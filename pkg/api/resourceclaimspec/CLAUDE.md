# Package: resourceclaimspec

## Purpose
Provides utilities for handling feature-gated fields in ResourceClaimSpec for Dynamic Resource Allocation (DRA).

## Key Functions
- `DropDisabledFields(new, old *resource.ResourceClaimSpec)` - Master function that removes disabled feature fields from ResourceClaimSpec

### Feature-Specific Drop Functions
- `dropDisabledDRADeviceTaintsFields` - Removes Tolerations from device requests when DRADeviceTaints feature is disabled
- `dropDisabledDRAPrioritizedListFields` - Removes FirstAvailable from requests when DRAPrioritizedList feature is disabled
- `dropDisabledDRAAdminAccessFields` - Removes AdminAccess from requests when DRAAdminAccess feature is disabled
- `dropDisabledDRAResourceClaimConsumableCapacityFields` - Removes DistinctAttribute from constraints and Capacity from requests when DRAConsumableCapacity feature is disabled

### Feature Detection Functions (exported)
- `DRAAdminAccessFeatureInUse(spec *resource.ResourceClaimSpec) bool` - Checks if admin access is in use
- `DRAConsumableCapacityFeatureInUse(spec *resource.ResourceClaimSpec) bool` - Checks if consumable capacity is in use

## Design Notes
- All functions preserve existing fields if they were already in use (to support updates to existing objects)
- The spec is immutable in practice, but drop functions handle the theoretical update case
