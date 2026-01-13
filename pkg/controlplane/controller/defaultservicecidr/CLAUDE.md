# Package: defaultservicecidr

## Purpose
This controller creates and manages the default ServiceCIDR resource named "kubernetes". The ServiceCIDR defines the IP range(s) used for allocating ClusterIPs to Services, derived from the `--service-cluster-ip-range` flag.

## Key Types

- **Controller**: Manages the "kubernetes" ServiceCIDR resource, creating it if missing and updating its status

## Key Functions

- **NewController()**: Creates a controller with primary and optional secondary (dual-stack) CIDR ranges
- **Start()**: Starts the controller, waits for initial sync, then runs periodic reconciliation
- **sync()**: Creates the ServiceCIDR if missing, handles single-to-dual-stack upgrades, syncs status
- **syncStatus()**: Sets the Ready condition on the ServiceCIDR when configuration matches

## Behavior

- Creates the "kubernetes" ServiceCIDR with configured CIDRs on startup
- Supports upgrading from single-stack to dual-stack by updating existing ServiceCIDR
- Sets Ready=True condition when CIDR values match controller configuration
- Does not modify ServiceCIDR if configuration doesn't match (requires admin intervention)
- Reports events for errors and configuration mismatches

## Design Notes

- Uses a filtered informer watching only the "kubernetes" ServiceCIDR
- Blocks API server startup until initial sync succeeds (polls every 100ms)
- Runs sync loop every 10 seconds (same as DefaultEndpointReconcilerInterval)
- Does not attempt to change Ready=False condition set by other components to avoid hotlooping
