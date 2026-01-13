# Package: rest

## Purpose
Implements the proxy subresource REST endpoint for Node objects, enabling HTTP proxying to node kubelets.

## Key Types

- **ProxyREST**: Implements the proxy subresource for a Node, allowing HTTP methods to be forwarded to the node's kubelet.

## Key Functions

- **New()**: Returns an empty NodeProxyOptions object.
- **ConnectMethods()**: Returns all HTTP methods that can be proxied: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS.
- **NewConnectOptions()**: Returns versioned proxy parameters with "path" as the subpath parameter.
- **Connect()**: Returns an HTTP handler that proxies requests to the node. Uses `node.ResourceLocation` to determine the target URL and wraps with `UpgradeAwareHandler` for WebSocket/SPDY upgrade support.
- **newThrottledUpgradeAwareProxyHandler()**: Creates a proxy handler with bandwidth throttling via `MaxBytesPerSec`.

## Design Notes

- Implements `rest.Connecter` interface for connect-style subresources.
- Supports all standard HTTP methods for maximum flexibility.
- Uses upgrade-aware proxy handler for WebSocket connections.
- Applies per-connection bandwidth limits from capabilities.
- Shared store with main REST handler (no explicit Destroy).
