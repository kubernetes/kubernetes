# Package: healthcheck

This package provides health checking functionality for kube-proxy, including both proxy-level health checks and per-service health checks for LoadBalancer services with externalTrafficPolicy: Local.

## Key Types

- `ProxyHealthServer` - HTTP server providing /healthz and /livez endpoints for kube-proxy
- `ServiceHealthServer` - Manages per-service health check endpoints for Local traffic policy
- `FakeServiceHealthServer` - Testing implementation

## Key Functions

- `NewProxyHealthServer()` - Creates a health server for overall proxy health
- `NewServiceHealthServer()` - Creates a server managing per-service health endpoints
- `SyncServices()` - Updates health check endpoints based on current services
- `SyncEndpoints()` - Updates endpoint counts for health responses

## Design Notes

- Proxy health (/healthz) returns 503 if last sync was too long ago
- Service health checks return 200 if local endpoints exist, 503 otherwise
- Used by cloud load balancers to determine which nodes should receive traffic
- Health check ports are specified per-service in the Service spec
- Supports graceful handling of endpoint changes during pod migrations
