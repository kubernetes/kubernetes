# Package: options

## Purpose
This package defines command-line flags and options for configuring a generic control plane API server. It aggregates options for etcd, authentication, authorization, admission, secure serving, and other server components.

## Key Types

- **Options**: Main struct aggregating all server options including GenericServerRunOptions, Etcd, SecureServing, Authentication, Authorization, Admission, etc. Also includes peer proxy settings and service account configuration
- **CompletedOptions**: Wrapper around validated and completed options that enforces Complete() is called before use

## Key Functions

- **NewOptions()**: Creates Options with sensible defaults (1-hour event TTL, protobuf storage format, system namespaces)
- **Options.AddFlags()**: Registers all command-line flags for the server components
- **Options.Complete()**: Validates and fills in defaults - sets advertise address, completes service account options, normalizes runtime config
- **Options.Validate()**: Runs validation on all component options and returns aggregated errors
- **ServiceIPRange()**: Validates and parses service cluster IP range, returning the API server service IP

## Key Flags

- `--event-ttl`: How long to retain events
- `--proxy-client-cert-file/key-file`: Client certs for aggregator/webhook calls
- `--peer-ca-file`, `--peer-advertise-ip/port`: Peer proxy configuration for multi-apiserver setups
- `--service-account-signing-key-file/endpoint`: Service account token signing configuration
- `--coordinated-leadership-*`: Leader election timing parameters

## Design Notes

- Uses the "completed options" pattern to ensure validation before use
- Supports both local key-based and external endpoint-based service account token signing
- Validates feature gate requirements for peer proxy and external JWT signer options
