# Package: plugin

## Purpose
This package implements exec-based credential provider plugins for Kubernetes image pulls. It allows external binaries to provide registry credentials dynamically, supporting service account token injection for workload identity scenarios.

## Key Types

- **pluginProvider**: Main provider that caches credentials and executes plugins
- **execPlugin**: Executes external credential provider binaries
- **perPodPluginProvider**: Wraps pluginProvider with per-pod context (namespace, name, service account)
- **serviceAccountProvider**: Handles service account token generation for plugins
- **cacheEntry**: Cached credentials with expiration time
- **Plugin**: Interface for executing credential provider plugins

## Key Functions

- **RegisterCredentialProviderPlugins()**: Registers plugins from CredentialProviderConfig file
- **NewExternalCredentialProviderDockerKeyring()**: Creates a keyring backed by external plugins
- **ExecPlugin()**: Executes plugin binary with CredentialProviderRequest, returns CredentialProviderResponse
- **provide()**: Gets credentials from cache or plugin, handles service account tokens
- **isImageAllowed()**: Checks if image matches plugin's matchImages patterns

## Plugin Protocol

- Plugin receives CredentialProviderRequest via stdin (JSON)
- Plugin returns CredentialProviderResponse via stdout
- Supports API versions: v1alpha1, v1beta1, v1
- 1-minute execution timeout for all plugins
- Cache key types: image, registry, or global

## Design Notes

- Uses singleflight to deduplicate concurrent requests for same image
- Credentials cached based on CacheDuration from plugin response
- Service account tokens bound to pods for workload identity
- Cache purged every 15 minutes for expired entries
- Plugins found in configurable pluginBinDir directory
- Supports environment variables and arguments per plugin
