# Package: framework

## Purpose
Provides fake implementations of scheduler framework interfaces for unit testing scheduler plugins and components.

## Key Types
- `FakeExtender` - Mock implementation of scheduler extender
- `FakeLister` - Mock implementation of pod/node listers
- `FakePlugin` - Configurable fake plugin for testing extension points

## Key Functions
- `NewFakeExtender()` - Creates a fake extender with configurable behavior
- `NewFakeLister()` - Creates fake listers with predefined objects
- Helper functions for constructing test scenarios

## Design Patterns
- Mock objects implement the same interfaces as production code
- Configurable return values and error injection
- Supports testing filter, score, and bind plugins
- Enables isolated unit testing without real cluster
