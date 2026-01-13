# Package: testing

Testing utilities for the runtimeclass package.

## Constants

- `SandboxRuntimeClass`: "sandbox" - pre-populated RuntimeClass name
- `SandboxRuntimeHandler`: "kata-containers" - handler for SandboxRuntimeClass
- `EmptyRuntimeClass`: "native" - RuntimeClass with empty handler

## Key Functions

- `NewPopulatedClient()`: Creates a fake Kubernetes client pre-populated with test RuntimeClasses (EmptyRuntimeClass and SandboxRuntimeClass).
- `StartManagerSync(m)`: Starts the manager, waits for cache sync, and returns a cleanup function. Usage: `defer StartManagerSync(t, m)()`
- `NewRuntimeClass(name, handler)`: Helper to create RuntimeClass objects.

## Design Notes

- Provides consistent test fixtures for RuntimeClass testing
- StartManagerSync handles proper lifecycle management with cleanup
- Fake client enables unit testing without real API server
