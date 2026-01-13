# Package: testing

## Purpose
Provides comprehensive test utilities, fake implementations, and mock objects for testing volume plugins.

## Key Types/Structs
- `FakeVolumeHost` - Mock implementation of VolumeHost for testing
- `fakeVolumeHost` - Internal implementation with configurable behaviors
- `FakeVolumePlugin` - Configurable fake volume plugin for tests
- `FakeBasicVolumePlugin` - Simplified fake plugin for basic tests

## Key Functions
- `NewFakeVolumeHost()` - Creates a fake volume host for testing
- `NewFakeVolumeHostWithCSINodeName()` - Creates fake host with CSI support
- `ProbeVolumePlugins()` - Returns test volume plugins
- Various assertion helpers for verifying volume operations

## Design Patterns
- Fake implementations of all major volume interfaces
- Configurable error injection for testing error paths
- Mock mounter/unmounter tracking for verification
- Thread-safe implementations for concurrent test execution
- Shared test infrastructure across volume plugin tests
