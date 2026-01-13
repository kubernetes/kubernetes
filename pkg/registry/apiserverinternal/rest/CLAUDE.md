# Package: rest

This package provides the REST storage provider for the internal.apiserver.k8s.io API group.

## Key Types

- `StorageProvider` - Implements genericapiserver.RESTStorageProvider

## Key Functions

- `NewRESTStorage()` - Creates storage for all apiserver internal resources
- `v1alpha1Storage()` - Returns v1alpha1 API version storage
- `GroupName()` - Returns "internal.apiserver.k8s.io"

## Resources Registered

- `storageversions` - Tracks which storage version is used for each resource
- `storageversions/status` - Status subresource for storage versions

## Design Notes

- Currently only contains StorageVersion resource
- StorageVersion helps coordinate storage migrations
- Used internally by the API server for version management
- v1alpha1 is the only version currently supported
