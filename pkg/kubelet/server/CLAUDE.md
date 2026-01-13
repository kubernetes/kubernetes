# Package: server

## Purpose
The `server` package implements the Kubelet's HTTP server, exposing Kubelet functionality over HTTP and gRPC. It handles incoming requests for pod management, container operations, metrics, logs, and debugging.

## Key Types/Structs

- **Server**: Main HTTP server struct that implements `http.Handler`. Contains auth interfaces, host interface, and restful container for routing.
- **TLSOptions**: Holds TLS configuration including certificates and keys.
- **AuthInterface**: Interface combining authentication, authorization, and node request attributes.
- **HostInterface**: Interface defining all kubelet methods required by the server (stats, container operations, logs, exec, attach, port-forward).
- **KubeletAuth**: Implementation of AuthInterface that combines authenticator, authorizer, and attribute getter.

## Key Functions

- **ListenAndServeKubeletServer**: Starts the main kubelet HTTP server with TLS support.
- **ListenAndServeKubeletReadOnlyServer**: Starts a read-only HTTP server without authentication.
- **ListenAndServePodResources**: Starts a gRPC server for the PodResources API.
- **NewServer**: Creates and configures a new Server with all handlers installed.
- **InstallAuthFilter**: Adds authentication/authorization filter to all requests.
- **InstallAuthNotRequiredHandlers**: Registers handlers that don't require auth (healthz, pods, stats, metrics).
- **InstallAuthRequiredHandlers**: Registers handlers requiring auth (exec, attach, portForward, containerLogs, checkpoint).

## HTTP Endpoints

- `/healthz` - Health checks
- `/pods` - List pods bound to the kubelet
- `/stats/summary` - Node and pod statistics
- `/metrics`, `/metrics/cadvisor`, `/metrics/resource`, `/metrics/probes` - Prometheus metrics
- `/exec`, `/attach`, `/portForward` - Container streaming operations
- `/containerLogs` - Container log retrieval
- `/logs` - System log access
- `/debug/pprof` - Profiling endpoints
- `/checkpoint` - Container checkpoint (feature-gated)

## Design Notes

- Uses go-restful for HTTP routing with filter-based authentication/authorization.
- Supports both secure (TLS) and read-only server modes.
- Authorization maps HTTP paths to Kubernetes RBAC subresources (nodes/stats, nodes/log, nodes/proxy, etc.).
- Long-running requests (exec, attach, portforward) are tracked separately for metrics.
