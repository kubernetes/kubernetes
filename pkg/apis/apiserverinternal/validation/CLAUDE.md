# Package: validation

## Purpose
Provides validation logic for StorageVersion resources in the apiserverinternal API group, ensuring consistency of storage version information across API servers.

## Key Functions
- `ValidateStorageVersion(sv *StorageVersion)`: Validates a complete StorageVersion object including metadata and status
- `ValidateStorageVersionName(name string, prefix bool)`: Validates storage version names follow the `<group>.<resource>` format
- `ValidateStorageVersionUpdate(sv, oldSV *StorageVersion)`: Validates updates (currently allows all updates)
- `ValidateStorageVersionStatusUpdate(sv, oldSV *StorageVersion)`: Validates status updates
- `validateStorageVersionStatus`: Validates status including server versions and conditions
- `validateServerStorageVersion`: Validates per-server version info ensuring encoding versions are in decodable versions
- `validateCommonVersion`: Ensures commonEncodingVersion matches actual common version across servers
- `isValidAPIVersion(apiVersion string)`: Validates API version format (group/version or just version)

## Validation Rules
- Storage version names must be in `<group>.<resource>` format with DNS-valid segments
- API server IDs must be unique within a StorageVersion
- Encoding version must be included in decodable versions
- Served versions must be included in decodable versions
- CommonEncodingVersion must match if all servers agree on encoding version
- Conditions must have unique types, and require reason and message fields
