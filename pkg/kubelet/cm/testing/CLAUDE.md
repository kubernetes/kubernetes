# Package testing

Package testing provides mock implementations of the ContainerManager interface for unit testing.

## Key Types

- `MockContainerManager`: Generated mock for ContainerManager interface

## Generated Mock

Uses mockery (github.com/vektra/mockery) to generate comprehensive mocks for:
- All ContainerManager methods
- All embedded interfaces (CPUsProvider, DevicesProvider, MemoryProvider, DynamicResourcesProvider)

## Mock Features

- Expecter pattern for fluent test setup
- Automatic test cleanup registration
- Return value chaining with Run() callbacks
- Argument matching with mock.Anything

## Design Notes

- Generated code (do not edit manually)
- Covers the full ContainerManager interface
- Used throughout kubelet unit tests
