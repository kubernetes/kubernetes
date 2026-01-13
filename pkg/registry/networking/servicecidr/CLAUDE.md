# Package: servicecidr

Implements the API server registry strategy for ServiceCIDR resources used for Service IP range management.

## Key Types

- **serviceCIDRStrategy**: Implements create/update/delete strategies for ServiceCIDR objects.
- **serviceCIDRStatusStrategy**: Separate strategy for status subresource updates.

## Key Functions

- **PrepareForCreate / PrepareForUpdate**: Minimal preparation (mostly placeholders).
- **Validate / ValidateUpdate**: Validates ServiceCIDR objects using networking validation.
- **GetResetFields**: Defines reset fields for networking/v1 and networking/v1beta1.
- **PrepareForUpdate (status)**: Preserves spec and resets metadata for status-only updates.

## Design Notes

- ServiceCIDR is a cluster-scoped resource defining IP ranges available for Service ClusterIPs.
- Part of the multi-CIDR Service allocation system.
- Has a status subresource for tracking allocation state.
- Supports both v1 and v1beta1 API versions.
