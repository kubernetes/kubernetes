# Package: pod

## Purpose
Provides the registry interface and REST strategy implementation for storing Pod API objects, including multiple subresource strategies (status, ephemeralContainers, resize) and URL location helpers.

## Key Types

- **podStrategy**: Main strategy for Pod CRUD operations.
- **podStatusStrategy**: Strategy for /status subresource.
- **podEphemeralContainersStrategy**: Strategy for /ephemeralcontainers subresource.
- **podResizeStrategy**: Strategy for /resize subresource (container resource updates).
- **ResourceGetter**: Interface for retrieving pods by name.

## Key Functions

- **Strategy, StatusStrategy, EphemeralContainersStrategy, ResizeStrategy** (vars): Strategies for different subresources.
- **NamespaceScoped()**: Returns `true` - Pods are namespace-scoped.
- **PrepareForCreate()**: Sets generation=1, status phase to Pending, QOS class, applies scheduling gates, mutates affinity/topology constraints, handles AppArmor version skew.
- **PrepareForUpdate()**: Preserves status, drops disabled fields, updates generation on spec changes.
- **CheckGracefulDelete()**: Implements graceful deletion with configurable grace period.
- **GetAttrs()**: Returns labels and selectable fields.
- **MatchPod()**: Returns selection predicate with nodeName indexing.
- **ToSelectableFields()**: Rich field selection including nodeName, restartPolicy, schedulerName, serviceAccountName, hostNetwork, phase, podIP, nominatedNodeName.
- **ResourceLocation()**: Returns URL for proxying to pod IP.
- **LogLocation()**: Returns kubelet URL for container logs.
- **AttachLocation(), ExecLocation()**: Returns URLs for attach/exec streaming.
- **PortForwardLocation()**: Returns URL for port forwarding.

## Design Notes

- Multiple strategies for different update paths (status, ephemeralContainers, resize).
- Rich field indexing for efficient queries by nodeName and namespace.
- Implements affinity/topology constraint mutation for matchLabelKeys.
- Handles AppArmor annotation-to-field migration.
- Generation tracking for pod spec changes.
- Graceful deletion with phase-aware grace period handling.
