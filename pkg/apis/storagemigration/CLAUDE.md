# Package: storagemigration

## Purpose
Defines internal (unversioned) API types for the storagemigration.k8s.io API group, which handles storage version migration for resources.

## Key Types/Structs
- `StorageVersionMigration`: Represents a migration of stored data to the current storage version
- `StorageVersionMigrationSpec`: Specifies which resource to migrate
- `StorageVersionMigrationStatus`: Tracks migration progress and conditions

## Design Notes
- This API group manages the migration of resources stored in etcd to their current storage version
- Used when API storage versions change and existing data needs to be updated
- Internal types are converted to/from versioned types (v1beta1) for external use
- Follows the standard Kubernetes internal API type pattern
