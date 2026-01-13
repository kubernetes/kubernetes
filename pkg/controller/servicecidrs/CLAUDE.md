# Package: servicecidrs

Implements the ServiceCIDR controller that manages service IP address ranges and their lifecycle.

## Key Types

- **Controller**: Manages ServiceCIDR resources and their finalizers.

## Key Functions

- **NewController**: Creates a new ServiceCIDR controller.
- **Run**: Starts workers processing ServiceCIDR events.
- **sync**: Main sync logic - handles finalizer management for ServiceCIDRs.
- **canDeleteCIDR**: Checks if a CIDR can be safely deleted (no IPs in use).
- **addServiceCIDRFinalizerIfNeeded**: Adds finalizer to protect CIDRs with allocated IPs.
- **removeServiceCIDRFinalizer**: Removes finalizer when CIDR is safe to delete.

## Key Constants

- **ServiceCIDRProtectionFinalizer**: `networking.k8s.io/service-cidr-finalizer`

## Design Patterns

- Uses finalizers to prevent deletion of ServiceCIDRs with allocated IPs.
- Checks IPAddress allocations before allowing CIDR deletion.
- Handles overlapping CIDRs correctly (only blocks deletion if no other CIDR contains the IPs).
- Maintains Ready condition on ServiceCIDR status.
- Uses workqueue with rate-limited retries.
