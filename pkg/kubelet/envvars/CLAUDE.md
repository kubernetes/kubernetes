# Package: envvars

Generates service-discovery environment variables for containers based on Kubernetes Services.

## Key Functions

- **FromServices(services []*v1.Service) []v1.EnvVar**: Main entry point that takes a list of Services and returns environment variables that tell containers how to find those services.

## Environment Variables Generated

For each service with a ClusterIP:
- `{SERVICE}_SERVICE_HOST`: The service's cluster IP
- `{SERVICE}_SERVICE_PORT`: The first port number
- `{SERVICE}_SERVICE_PORT_{NAME}`: Named port numbers

Docker-compatible link variables:
- `{SERVICE}_PORT`: Full URL (e.g., `tcp://10.0.0.1:80`)
- `{SERVICE}_PORT_{PORT}_{PROTO}`: URL for specific port
- `{SERVICE}_PORT_{PORT}_{PROTO}_PROTO`: Protocol (tcp/udp)
- `{SERVICE}_PORT_{PORT}_{PROTO}_PORT`: Port number
- `{SERVICE}_PORT_{PORT}_{PROTO}_ADDR`: IP address

## Helper Functions

- `makeEnvVariableName()`: Converts service names to valid env var names (uppercase, dashes to underscores)
- `makeLinkVariables()`: Generates Docker-compatible service link variables

## Notes

- Services without ClusterIP (headless services) are skipped
- Follows Docker linking conventions for backwards compatibility
