# Package: testutil

Provides testing utilities and fake implementations for controller unit tests.

## Key Types

- **FakeNodeHandler**: Fake implementation of NodeInterface for testing node operations.
- **FakeLegacyHandler**: Fake CoreV1Interface wrapping FakeNodeHandler.
- **FakeRecorder**: Fake event recorder that captures events for test assertions.

## Key Functions (FakeNodeHandler)

- **Create/Get/List/Delete/Update**: Node CRUD operations with tracking.
- **UpdateStatus**: Updates node status with tracking.
- **Patch/PatchStatus**: Applies patches to nodes.
- **GetUpdatedNodesCopy**: Returns copy of all updated nodes.

## Key Functions (Helpers)

- **NewNode**: Creates a test Node with default capacity.
- **NewPod**: Creates a test Pod assigned to a host.
- **NewFakeRecorder**: Creates a fake event recorder.
- **GetKey**: Gets the workqueue key for a Kubernetes object.
- **GetZones**: Returns list of zones from nodes in FakeNodeHandler.
- **CreateZoneID**: Creates a zone identifier from region and zone.

## Design Patterns

- FakeNodeHandler tracks created, deleted, and updated nodes separately.
- Supports async operations via AsyncCalls slice.
- Provides wait channels (DeleteWaitChan, PatchWaitChan) for synchronization.
- FakeRecorder stores events in slice for test verification.
- Used extensively by nodelifecycle, tainteviction, and other controller tests.
