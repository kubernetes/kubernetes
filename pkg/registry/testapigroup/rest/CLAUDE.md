# Package: rest

## Purpose
Provides the RESTStorageProvider for the testapigroup.apimachinery.k8s.io API group, which is used for testing API server functionality.

## Key Types

- **RESTStorageProvider**: Implements genericapiserver.RESTStorageProvider for the test API group.

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter)**: Creates APIGroupInfo with storage for test API versions.
- **v1Storage(...)**: Creates storage map for v1 resources (Carp).
- **GroupName()**: Returns "testapigroup.apimachinery.k8s.io".

## Registered Resources

- **v1**: carps, carps/status

## Design Notes

- Used for testing API server registry implementations.
- Not intended for production use.
- Demonstrates typical patterns for registering API resources.
