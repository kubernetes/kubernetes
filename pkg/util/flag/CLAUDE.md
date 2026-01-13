# Package: flag

## Purpose
Provides custom pflag.Value implementations for validating and parsing complex command-line flag types used in Kubernetes components.

## Key Types
- `IPVar` - Validates IP address flags
- `IPPortVar` - Validates IP or IP:port flags
- `PortRangeVar` - Validates port range flags (e.g., "30000-32767")
- `ReservedMemoryVar` - Parses NUMA node memory reservations
- `RegisterWithTaintsVar` - Parses node taints for registration

## Key Functions
- `Set()` - Parses and validates flag value (implements pflag.Value)
- `String()` - Returns string representation of current value
- `Type()` - Returns type name for help text

## Supported Formats
- IP: "192.168.1.1" or "::1"
- IPPort: "192.168.1.1:8080" or just "192.168.1.1"
- PortRange: "30000-32767"
- ReservedMemory: "0:memory=1Gi;1:memory=2Gi,hugepages-2Mi=512Mi"
- Taints: "key=value:NoSchedule,key2:NoExecute"

## Design Patterns
- Implements pflag.Value interface for integration with cobra/pflag
- Validation on Set() with descriptive error messages
- Pointer-based storage for integration with config structs
