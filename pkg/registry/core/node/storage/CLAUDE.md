# Package: storage

## Purpose
Provides REST storage implementation for Node objects including main resource, status subresource, and proxy subresource with kubelet connection handling.

## Key Types

- **NodeStorage**: Container holding Node, Status, and Proxy REST handlers plus kubelet connection info.
- **REST**: Main storage for Node operations, implements `rest.Redirector`.
- **StatusREST**: Storage for /status subresource updates.

## Key Functions

- **NewStorage(optsGetter, kubeletClientConfig, proxyTransport)**: Creates NodeStorage with all subresources. Notable features:
  - Sets up kubelet connection info getter for node-to-kubelet communication
  - Creates NodeGetterFunc to retrieve nodes and convert to v1.Node
  - Configures proxy transport for node proxy requests

- **ResourceLocation()**: Implements `rest.Redirector` - returns URL for traffic to specified node.

- **ShortNames()**: Returns `["no"]` for kubectl.

## Design Notes

- Implements `rest.Redirector` for node resource location.
- Creates `NodeConnectionInfoGetter` for kubelet communication.
- Converts internal api.Node to v1.Node for external clients.
- Shares underlying store between REST types.
- Status subresource uses different update strategy (node.StatusStrategy).
