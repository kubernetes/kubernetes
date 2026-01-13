# Package testing

Package testing provides mock implementations of the podresources provider interfaces for unit testing.

## Key Types

- `MockDevicesProvider`: Mock for DevicesProvider interface
- `MockPodsProvider`: Mock for PodsProvider interface
- `MockCPUsProvider`: Mock for CPUsProvider interface
- `MockMemoryProvider`: Mock for MemoryProvider interface
- `MockDynamicResourcesProvider`: Mock for DynamicResourcesProvider interface

## Constructor Functions

- `NewMockDevicesProvider(t)`: Creates a new MockDevicesProvider with test cleanup
- `NewMockPodsProvider(t)`: Creates a new MockPodsProvider with test cleanup
- `NewMockCPUsProvider(t)`: Creates a new MockCPUsProvider with test cleanup
- `NewMockMemoryProvider(t)`: Creates a new MockMemoryProvider with test cleanup
- `NewMockDynamicResourcesProvider(t)`: Creates a new MockDynamicResourcesProvider with test cleanup

## Design Notes

- Generated using mockery tool (github.com/vektra/mockery)
- Uses testify/mock for mock implementation
- Each mock includes an Expecter for fluent mock setup
- Automatically registers cleanup and assertion functions with the test
