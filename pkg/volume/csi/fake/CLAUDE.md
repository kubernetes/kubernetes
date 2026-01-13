# Package: fake

## Purpose
Provides fake/mock implementations of CSI driver clients for testing CSI volume plugin functionality.

## Key Types/Structs
- `FakeClient` - Mock CSI driver client implementing csi.NodeClient interface
- `FakeNodeClient` - Fake node service client for NodeStage/NodePublish operations
- `FakeCloser` - Mock gRPC connection closer

## Key Functions
- `NewClient()` - Creates a new fake CSI client for testing
- `NodePublishVolume()` - Mock implementation of CSI NodePublishVolume
- `NodeUnpublishVolume()` - Mock implementation of CSI NodeUnpublishVolume
- `NodeGetCapabilities()` - Returns configurable node capabilities

## Design Patterns
- Allows injection of errors and custom responses for testing
- Tracks call counts and arguments for verification in tests
- Configurable capability responses for testing different CSI driver behaviors
