# Package: rest

## Purpose
Provides the RESTStorageProvider for the storagemigration.k8s.io API group, which wires up storage migration resources to the API server.

## Key Types

- **RESTStorageProvider**: Implements genericapiserver.RESTStorageProvider for the storagemigration API group.

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage for storagemigration API versions.
- **v1beta1Storage(...)**: Creates storage map for v1beta1 resources (StorageVersionMigration).
- **GroupName()**: Returns "storagemigration.k8s.io".

## Registered Resources

- **v1beta1**: storageversionmigrations, storageversionmigrations/status

## Feature Gating

- **StorageVersionMigrator**: Must be enabled for storage to be registered. Logs warning if disabled.

## Design Notes

- StorageVersionMigration enables migrating stored data between API versions.
- Controlled by the StorageVersionMigrator feature gate for gradual rollout.
