# Package: rest

Provides the REST storage provider for the `events.k8s.io` API group.

## Key Types

- **RESTStorageProvider**: Implements `genericapiserver.RESTStorageProvider` with a configurable TTL for event storage.

## Key Functions

- **NewRESTStorage**: Creates the APIGroupInfo for events API resources.
- **v1Storage**: Configures storage for Events with TTL-based expiration.
- **GroupName**: Returns "events.k8s.io".

## Design Notes

- Events have a configurable TTL (time-to-live) for automatic cleanup.
- Reuses the core event storage implementation from `pkg/registry/core/event/storage`.
- The events.k8s.io API group provides the newer Event API (vs core/v1 Events).
