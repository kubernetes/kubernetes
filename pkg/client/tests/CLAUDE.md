# Package: client/tests

## Purpose
Contains integration tests for the Kubernetes client libraries that require internal client access.

## Test Files
- `clientset_test.go`: Tests for clientset functionality
- `fake_client_test.go`: Tests for fake client implementations
- `listwatch_test.go`: Tests for list/watch operations
- `portfoward_test.go`: Tests for port forwarding functionality
- `remotecommand_test.go`: Tests for remote command execution (exec/attach)

## Design Notes
- Tests are in a separate package to avoid circular dependencies with internal types
- Uses internal client types that are not available in external client packages
- Provides coverage for client-side operations like portforward and remote command execution
