# Package: service

## Purpose
Provides Windows service integration for Kubernetes components, allowing kubelet and other components to run as Windows services with proper lifecycle management.

## Key Types/Structs
- `handler` - Windows service control handler
- `PreshutdownHandler` - Interface for handling preshutdown events
- `SERVICE_PRESHUTDOWN_INFO` - Windows structure for preshutdown timeout

## Key Functions
- `InitService()` - Initializes component as Windows service
- `InitServiceWithShutdown()` - Initializes with preshutdown event support
- `SetPreShutdownHandler()` - Registers handler for graceful shutdown
- `IsServiceInitialized()` - Checks if running as Windows service
- `QueryPreShutdownInfo()` - Queries preshutdown timeout configuration
- `UpdatePreShutdownInfo()` - Sets preshutdown timeout

## Design Patterns
- Windows-only implementation (build tag: windows)
- Integrates with Windows Service Control Manager (SCM)
- Supports graceful shutdown via SERVICE_CONTROL_PRESHUTDOWN
- Translates Windows service stop signals to Kubernetes shutdown signals
- Uses golang.org/x/sys/windows/svc for service interaction
- Global handler pattern (one service per process)
