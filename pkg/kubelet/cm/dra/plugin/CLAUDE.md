# Package plugin

Package plugin provides the DRA (Dynamic Resource Allocation) plugin client for communicating with DRA drivers via gRPC.

## Key Types

- `DRAPlugin`: Represents a registered DRA driver plugin with gRPC connection
- `DRAPluginManager`: Manages registered DRA plugins (in dra_plugin_manager.go)

## DRAPlugin Methods

- `DriverName()`: Returns the driver name
- `NodePrepareResources(ctx, req)`: Calls driver to prepare resources for claims
- `NodeUnprepareResources(ctx, req)`: Calls driver to unprepare resources
- `NodeWatchResources(ctx)`: Establishes health monitoring stream
- `SetHealthStream/HealthStreamCancel`: Manages health stream lifecycle

## Supported API Versions

- `drapbv1.DRAPluginService`: v1 API (preferred)
- `drapbv1beta1.DRAPluginService`: v1beta1 API (legacy)

## Constants

- `defaultClientCallTimeout`: 45 seconds (half of kubelet retry period)

## gRPC Operations

NodePrepareResources:
- Input: List of claims to prepare
- Output: Per-claim results with CDI device IDs

NodeUnprepareResources:
- Input: List of claims to unprepare
- Output: Per-claim success/error status

## Design Notes

- Auto-negotiates API version with driver
- Metrics interceptor tracks gRPC operation duration
- Health streaming for device health updates
- Thread-safe health stream management
