# Package: resourceclaim

## Purpose
Implements the registry strategy for ResourceClaim objects in the DRA API. ResourceClaim is a namespaced resource that represents a request for device resources.

## Key Types

- **resourceclaimStrategy**: Implements REST strategy for ResourceClaim spec operations.
- **resourceclaimStatusStrategy**: Extends resourceclaimStrategy for status subresource operations.

## Key Functions

- **NewStrategy(nsClient)**: Creates strategy with namespace client for admin access validation.
- **NewStatusStrategy(resourceclaimStrategy)**: Creates status strategy wrapping the main strategy.
- **NamespaceScoped()**: Returns true - ResourceClaim is namespaced.
- **GetResetFields()**: Returns fields to reset per API version (status for spec updates, spec/metadata for status updates).
- **Match(label, field)**: Returns a SelectionPredicate for filtering ResourceClaims.
- **GetAttrs(obj)**: Returns labels and selectable fields for a ResourceClaim.

## Feature Gating

Handles multiple feature gates with field dropping when disabled:
- **DRAResourceClaimDeviceStatus**: Controls status.Devices field
- **DRAAdminAccess**: Controls adminAccess in allocation results
- **DRAConsumableCapacity**: Controls shareID and consumedCapacity fields
- **DRADeviceBindingConditions**: Controls bindingConditions fields

## Design Notes

- Validates admin access against namespace labels using AuthorizedForAdmin/AuthorizedForAdminStatus.
- Automatically cleans up stale device status entries after deallocation.
