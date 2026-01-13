# Package client

Package client provides configuration and transport utilities for connecting to kubelets from the API server or other control plane components.

## Key Types

- `KubeletClientConfig`: Configuration for kubelet client including port, TLS settings, timeout, and egress selector
- `KubeletTLSConfig`: TLS configuration (cert, key, CA files)
- `ConnectionInfo`: Connection details for a specific kubelet (scheme, hostname, port, transport)
- `ConnectionInfoGetter`: Interface for obtaining connection info by node name
- `NodeConnectionInfoGetter`: Implementation that looks up connection info from Node API objects
- `NodeGetter`: Interface for fetching Node objects
- `NodeGetterFunc`: Function adapter for NodeGetter

## Key Functions

- `MakeTransport`: Creates a secure HTTP RoundTripper for kubelet communication
- `MakeInsecureTransport`: Creates an insecure HTTP RoundTripper (skips TLS verification)
- `NewNodeConnectionInfoGetter`: Creates a ConnectionInfoGetter from node lister and config
- `GetConnectionInfo`: Retrieves kubelet connection info from Node status

## Design Notes

- Supports egress selector for network routing through API server
- Uses node's DaemonEndpoints.KubeletEndpoint.Port or falls back to default
- Respects preferred address types when selecting node address
- TLS files are reloaded automatically for certificate rotation
