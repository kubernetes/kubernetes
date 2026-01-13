# Package: internalversion

## Purpose
The `internalversion` package provides print handlers for all core Kubernetes resource types, converting internal API objects to human-readable table rows.

## Key Functions

- **AddHandlers**: Registers all print handlers with a PrintHandler.

## Supported Resources

Provides table printers for all major Kubernetes resources including:
- Pods, PodTemplates, ReplicationControllers
- Services, Endpoints, EndpointSlices
- Nodes, Namespaces, Events
- Deployments, ReplicaSets, StatefulSets, DaemonSets
- Jobs, CronJobs
- ConfigMaps, Secrets
- PersistentVolumes, PersistentVolumeClaims
- StorageClasses, VolumeAttachments
- Ingresses, NetworkPolicies
- ServiceAccounts, RBAC resources
- HorizontalPodAutoscalers
- PriorityClasses, ResourceQuotas, LimitRanges
- And many more...

## Design Notes

- Each resource type has column definitions and a print function.
- Print functions extract relevant fields for display.
- Wide output includes additional columns (marked with Priority > 0).
- Handles both single objects and lists.
- Uses internal API types (not versioned v1 types).
- Imported by API server for table conversion.
