# Package: apiserverinternal

## Purpose
Defines internal types for tracking storage versions across API server instances, enabling coordinated storage schema migrations.

## Key Types

### StorageVersion
Tracks storage encoding version for a specific resource:
- Name format: `<group>.<resource>` (e.g., "apps.deployments")
- `Spec` - Empty (placeholder for API conventions)
- `Status` - Contains version information from all API servers

### StorageVersionStatus
Contains:
- `StorageVersions` - Per-server version reports
- `CommonEncodingVersion` - Set when all servers agree on encoding version
- `Conditions` - Observable state conditions

### ServerStorageVersion
Per-API server report:
- `APIServerID` - Unique server identifier
- `EncodingVersion` - Version used when persisting to etcd
- `DecodableVersions` - All versions this server can read
- `ServedVersions` - All versions this server can serve

### StorageVersionCondition
Condition with:
- `Type` - e.g., AllEncodingVersionsEqual
- `Status` - True, False, Unknown
- `Reason` / `Message` - Human-readable details

## Use Cases
- Coordinating storage migrations across HA API server deployments
- Ensuring all API servers can decode objects before changing encoding version
- Detecting version skew between API servers
