# Package: rest

## Purpose
Provides the legacyProvider that wires together all core API group resources and registers them with the API server.

## Key Types

- **legacyProvider**: Implements RESTStorageProvider for core API group.
- **RESTStorageProvider**: Interface for providing REST storage implementations.

## Key Functions

- **NewRESTStorage(apiResourceConfigSource, restOptionsGetter, ...)**: Creates storage for all core resources.
- **GroupName()**: Returns "" (empty string for core/legacy API group).
- **NewLegacyRESTStorage()**: Main factory function creating all core resource storage:
  - Pods (with all subresources: binding, eviction, status, log, exec, attach, portforward, proxy)
  - Services (with IP and port allocation)
  - Nodes, Namespaces, Events
  - ConfigMaps, Secrets
  - PersistentVolumes, PersistentVolumeClaims
  - ReplicationControllers, ResourceQuotas
  - LimitRanges, Endpoints, ServiceAccounts
  - PodTemplates

## Design Notes

- Central wiring point for all core/v1 API resources.
- Configures kubelet connection for pod subresources.
- Sets up IP allocators for Services (ClusterIP, NodePort).
- Handles feature-gated resources and subresources.
- Integrates with PodDisruptionBudget client for eviction.
- Creates proxy transport with TLS configuration.
