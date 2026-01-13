# Package: fsquota

## Purpose
Provides filesystem quota management for volume size limiting, using project quotas on supported filesystems (XFS, ext4).

## Key Types/Structs
- `Interface` - Quota management interface
- `QuotaID` - Identifier for filesystem quotas

## Key Functions
- `SupportsQuotas()` - Checks if path supports filesystem quotas
- `AssignQuota()` - Assigns a quota to a directory path
- `GetConsumption()` - Gets current space usage via quota
- `GetInodes()` - Gets current inode usage via quota
- `ClearQuota()` - Removes quota from a path
- `GetQuotaOnDir()` - Gets quota ID for a directory

## Design Patterns
- Feature gated via LocalStorageCapacityIsolationFSQuotaMonitoring
- Uses XFS project quotas on Linux
- Faster than walking directory tree for usage calculation
- Platform-specific: full support on Linux, stubs elsewhere
- Manages project ID allocation and /etc/projects, /etc/projid files
