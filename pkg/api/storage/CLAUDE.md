# Package: storage

## Purpose
Provides warning generation for storage-related API objects (StorageClass, CSIStorageCapacity) focusing on deprecated node label usage.

## Key Functions
- `GetWarningsForStorageClass(sc *storage.StorageClass) []string` - Generates warnings for deprecated node labels used in allowedTopologies matchLabelExpressions
- `GetWarningsForCSIStorageCapacity(csc *storage.CSIStorageCapacity) []string` - Generates warnings for deprecated node labels in nodeTopology selector

## Warnings Generated
Both functions check for deprecated node labels such as:
- `beta.kubernetes.io/arch` and `beta.kubernetes.io/os`
- `failure-domain.beta.kubernetes.io/region` and `zone`
- `beta.kubernetes.io/instance-type`
- `node-role.kubernetes.io/master`

## Design Notes
- Delegates to `nodeapi.GetNodeLabelDeprecatedMessage` and `nodeapi.GetWarningsForNodeSelector` for consistent deprecation message handling across the codebase
