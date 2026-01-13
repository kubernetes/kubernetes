# Package: rest

Provides the REST storage provider for the `policy` API group.

## Key Types

- **RESTStorageProvider**: Implements `genericapiserver.RESTStorageProvider` for policy resources.

## Key Functions

- **NewRESTStorage**: Creates APIGroupInfo with policy resource storage.
- **v1Storage**: Configures v1 API storage for PodDisruptionBudgets (with status subresource).
- **GroupName**: Returns "policy".

## Design Notes

- Acts as the entry point for registering policy API resources with the API server.
- Currently only exposes PodDisruptionBudgets; add new policy resources here.
- Storage is conditionally created based on APIResourceConfigSource settings.
- PodDisruptionBudget includes a status subresource for disruption tracking.
