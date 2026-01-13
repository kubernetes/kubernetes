# Package: resource

## Purpose
Provides utility functions for validating admin access authorization for Dynamic Resource Allocation (DRA) device requests. This package contains shared validation logic used by resource registry implementations.

## Key Functions

- **AuthorizedForAdmin(ctx, deviceRequests, namespaceName, nsClient)**: Checks if a request is authorized for admin access to devices based on the namespace label. Returns field.ErrorList with any authorization errors.

- **AuthorizedForAdminStatus(ctx, newAllocationResult, oldAllocationResult, namespaceName, nsClient)**: Validates admin access authorization for status updates. Checks if namespace has the required DRA admin label when admin access is requested in allocation results.

## Design Notes

- Admin access requires the namespace to have the `resource.DRAAdminNamespaceLabelKey: true` label.
- Spec validation is simpler since spec is immutable after creation.
- Status validation handles the case where old allocation already has admin access (immutable allocation).
- These functions are used by resourceclaim and resourceclaimtemplate strategies.
