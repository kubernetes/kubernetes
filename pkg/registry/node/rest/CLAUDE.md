# Package: rest

Provides the REST storage provider for the `node.k8s.io` API group.

## Key Types

- **RESTStorageProvider**: Implements `genericapiserver.RESTStorageProvider` for node API resources.

## Key Functions

- **NewRESTStorage**: Creates APIGroupInfo with node resource storage (currently RuntimeClass only).
- **v1Storage**: Configures v1 API storage for RuntimeClasses.
- **GroupName**: Returns "node.k8s.io".

## Design Notes

- Acts as the entry point for registering node API resources with the API server.
- Currently only exposes RuntimeClass; add new node resources here.
- Storage is conditionally created based on APIResourceConfigSource settings.
