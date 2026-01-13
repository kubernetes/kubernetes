# Package v1beta1

Package v1beta1 implements the v1beta1 Device Plugin gRPC API for communication between kubelet and device plugins.

## Key Types

- `RegistrationHandler`: Interface for handling plugin registration and directory cleanup
- `ClientHandler`: Interface for handling plugin connections and ListAndWatch responses

## Handler Interfaces

RegistrationHandler:
- `CleanupPluginDirectory(logger, string) error`: Cleans up plugin socket directory

ClientHandler:
- `PluginConnected(ctx, resourceName, DevicePlugin) error`: Called when plugin connects
- `PluginDisconnected(logger, resourceName)`: Called when plugin disconnects
- `PluginListAndWatchReceiver(logger, resourceName, response)`: Receives device updates

## Error Constants

- `errFailedToDialDevicePlugin`: Plugin unreachable on socket
- `errUnsupportedVersion`: API version mismatch
- `errInvalidResourceName`: Invalid resource name format
- `errBadSocket`: Non-absolute socket path

## Sub-package Components

- `server.go`: Registration gRPC server
- `client.go`: Device plugin gRPC client
- `handler.go`: Registration and client handler implementations
- `stub.go`: Testing stub implementation

## Design Notes

- Plugins register by creating Unix socket and calling Register RPC
- ListAndWatch streams device updates (healthy/unhealthy)
- Allocate RPC called during pod scheduling for device setup
