# Package: testing

## Purpose
Provides test utilities and wrappers for scheduler testing, including mock implementations and helper functions.

## Key Types
- `wrappers.PodWrapper` - Fluent builder for creating test Pod objects
- `wrappers.NodeWrapper` - Fluent builder for creating test Node objects

## Key Functions
- `MakePod()` - Creates a PodWrapper for building test pods
- `MakeNode()` - Creates a NodeWrapper for building test nodes
- Helper methods for setting pod/node attributes (labels, resources, etc.)

## Design Patterns
- Fluent/builder pattern for constructing test objects
- Chainable method calls for concise test setup
- Sensible defaults with override capabilities
- Reduces boilerplate in scheduler unit tests
