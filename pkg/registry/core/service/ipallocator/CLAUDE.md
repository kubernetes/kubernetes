# Package: ipallocator

## Purpose
Provides IP address allocation for Service ClusterIPs using IPAddress API objects as the allocation backend.

## Key Types

- **Interface**: IP allocator interface with Allocate, AllocateNext, Release, ForEach, Has, CIDR, IPFamily, DryRun, EnableMetrics.
- **Allocator**: Main implementation using IPAddress objects via networking.k8s.io API.
- **dryRunAllocator**: Shim for testing allocations without persisting.

## Key Functions

- **NewIPAllocator(cidr, client, ipAddressInformer)**: Creates allocator for a CIDR range using IPAddress API.
- **Allocate(ip)** / **AllocateService(svc, ip)**: Reserves specific IP address.
- **AllocateNext()** / **AllocateNextService(svc)**: Allocates next available IP using random scan with range offset.
- **Release(ip)**: Deletes IPAddress object to free the IP.
- **createIPAddress()**: Creates IPAddress object with ParentRef to owning Service.
- **ipIterator()**: Creates iterator that scans IP range from random offset.

## Design Notes

- Uses IPAddress custom resources (networking.k8s.io/v1) instead of bitmap in etcd.
- KEP-3070: Subdivides range with offset to prefer dynamic allocation from upper range.
- Supports IPv4 and IPv6 (minimum /64 prefix for IPv6).
- Metrics support for allocation tracking.
- Handles mid-air collisions (TOCTOU races) by retrying.
