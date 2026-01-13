# Package: rest

Provides the REST storage provider for the `discovery.k8s.io` API group, wiring up all discovery resources.

## Key Types

- **StorageProvider**: Implements `genericapiserver.RESTStorageProvider` to create and configure storage for discovery API resources.

## Key Functions

- **NewRESTStorage**: Creates the APIGroupInfo containing versioned storage maps for discovery resources.
- **v1Storage**: Configures storage for v1 API version, currently only EndpointSlices.
- **GroupName**: Returns "discovery.k8s.io".

## Design Notes

- Acts as the entry point for registering discovery API resources with the API server.
- Currently only exposes EndpointSlices; add new discovery resources here.
- Storage is conditionally created based on `APIResourceConfigSource` feature enablement.
