# Package: rest

## Purpose
Implements REST endpoints for Pod subresources: log streaming, proxy, exec, attach, and port-forward operations.

## Key Types

- **LogREST**: Implements /log endpoint for streaming container logs.
- **ProxyREST**: Implements /proxy endpoint for HTTP proxying to pod.
- **AttachREST**: Implements /attach endpoint for attaching to container stdin/stdout.
- **ExecREST**: Implements /exec endpoint for executing commands in containers.
- **PortForwardREST**: Implements /portforward endpoint for port forwarding.

## Key Functions

- **LogREST.Get()**: Validates log options, returns LocationStreamer for log content. Supports follow, previous, timestamps, sinceSeconds/Time, tailLines, limitBytes, and stream selection.

- **ProxyREST.Connect()**: Returns handler for proxying all HTTP methods to pod.

- **AttachREST.Connect()**: Returns handler for WebSocket/SPDY attach with optional stream translation for V5 protocol.

- **ExecREST.Connect()**: Returns handler for WebSocket/SPDY exec with command execution and optional stream translation.

- **PortForwardREST.Connect()**: Returns handler for port forwarding with WebSocket tunneling support.

- **newThrottledUpgradeAwareProxyHandler()**: Creates bandwidth-throttled upgrade-aware proxy handler.

## Design Notes

- All connect handlers implement `rest.Connecter` interface.
- Feature-gated authorization check for "create" verb on WebSocket upgrades (AuthorizePodWebsocketUpgradeCreatePermission).
- TranslateStreamCloseWebsocketRequests feature enables V5 WebSocket protocol translation.
- PortForwardWebsockets feature enables WebSocket-based tunneling.
- LogREST produces "text/plain" MIME type and tracks TLS skip metrics.
- Bandwidth limiting via capabilities.PerConnectionBandwidthLimitBytesPerSec.
