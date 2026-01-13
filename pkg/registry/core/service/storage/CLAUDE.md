# Package: storage

## Purpose
Provides REST storage implementation for Service objects with integrated IP and port allocation, status subresource, and proxy support.

## Key Types

- **REST**: Main storage with IP/port allocation, proxy transport, endpoints/pods references.
- **StatusREST**: Storage for /status subresource updates.
- **Allocators**: Container for IP allocators (per family) and port allocator.
- **Before/After**: Type wrappers to enforce correct argument ordering in functions.

## Key Functions

- **NewREST(optsGetter, serviceIPFamily, ipAllocs, portAlloc, endpoints, pods, proxyTransport)**: Creates REST, StatusREST, and ProxyREST.
- **beginCreate()**: Allocates IPs and ports transactionally before create.
- **beginUpdate()**: Handles allocation changes during updates.
- **afterDelete()**: Releases allocated resources and deletes endpoints.
- **defaultOnRead()**: Sets IPFamilies/IPFamilyPolicy for legacy Services.
- **normalizeClusterIPs()**: Syncs ClusterIP and ClusterIPs fields.
- **patchAllocatedValues()**: Preserves allocated values during idempotent updates.
- **ResourceLocation()**: Returns URL for proxying to service endpoints.
- **ShortNames()**: Returns `["svc"]`.
- **Categories()**: Returns `["all"]`.

## Design Notes

- Complex allocation logic with transactional commit/revert.
- Supports dual-stack (IPv4/IPv6) with primary/secondary IP families.
- Read-time defaulting for IPFamilies/IPFamilyPolicy on legacy Services.
- Handles ClusterIP/ClusterIPs synchronization for old clients.
- Implements rest.Redirector for proxy support.
