# Package: test

Provides testing utilities for the node IPAM controller.

## Key Functions

- **MustParseCIDR**: Parses a CIDR string or panics; useful for test setup.
- **FakeNodeInformer**: Creates a fake node informer populated with test nodes from a FakeNodeHandler.
- **WaitForUpdatedNodeWithTimeout**: Polls until a specified number of node updates have been processed.

## Key Variables

- **NodePollInterval**: Default polling interval (10ms) for waiting on node updates.
- **AlwaysReady**: A function that always returns true, used as a ready check in tests.

## Design Patterns

- Provides test doubles (fakes) for Kubernetes informers.
- Uses polling-based waiting for async test assertions.
- Designed to work with testutil.FakeNodeHandler for controller testing.
