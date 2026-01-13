# Package: authorizer

## Purpose
This package configures the authorization chain for the Kubernetes API server. It creates a union authorizer from multiple authorization modes and supports dynamic reloading of authorization configuration.

## Key Types

- **Config**: Authorization configuration including policy file, webhook settings, and informers
- **reloadableAuthorizerResolver**: Authorizer wrapper supporting dynamic configuration reload

## Key Functions

- **Config.New()**: Creates the authorizer chain based on AuthorizationConfiguration
- **LoadAndValidateFile()**: Loads and validates authorization config from file
- **LoadAndValidateData()**: Validates authorization configuration data
- **GetNameForAuthorizerMode()**: Returns the canonical name for an authorization mode

## Supported Authorization Modes

- **Node**: Authorizes kubelet API requests based on node identity
- **ABAC**: Attribute-Based Access Control using policy file
- **RBAC**: Role-Based Access Control using cluster roles/bindings
- **Webhook**: External webhook authorization
- **AlwaysAllow/AlwaysDeny**: Simple allow/deny modes

## Design Notes

- Authorizers are built once and persist across reloads (except Webhook)
- Node authorizer builds a graph from pods, PVs, and volume attachments
- Configuration file can be reloaded every minute without restart
- Non-webhook authorizer types must remain in config on reload
- Uses CEL compiler for authorization expression evaluation
