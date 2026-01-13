# Package: http

## Purpose
The `http` package implements HTTP probes for container health checks.

## Key Interfaces

- **Prober**: Interface for HTTP probes.
  - `Probe(req, timeout)`: Performs HTTP request and returns result.

- **GetHTTPInterface**: Interface for making HTTP requests.
  - `Do(req)`: Executes HTTP request.

## Key Functions

- **New**: Creates Prober with TLS verification disabled.
- **NewWithTLSConfig**: Creates Prober with custom TLS config.
- **DoHTTPProbe**: Performs HTTP probe (exported for direct use).
- **RedirectChecker**: Returns redirect policy function.

## Behavior

1. Sends HTTP request with configured timeout.
2. Returns Success for status 200-299.
3. Returns Warning for redirects (300-399) when not following.
4. Returns Failure for status >= 400 or connection errors.
5. Reads response body up to 10KB.

## Configuration

- **followNonLocalRedirects**: Whether to follow redirects to different hosts.
- Disables keep-alives and compression.
- Bypasses node's local proxy settings.

## Design Notes

- TLS verification disabled by default for internal probes.
- Response body truncated at 10KB to prevent memory issues.
- Redirect behavior configurable for security.
- Used by kubelet for HTTP-based liveness/readiness probes.
