# Package: credentialprovider

## Purpose
This package provides Docker registry credential management for Kubernetes, implementing a keyring system that looks up credentials for container image pulls. It supports reading credentials from .dockercfg and .docker/config.json files.

## Key Types

- **DockerKeyring**: Interface for looking up credentials by image name
- **BasicDockerKeyring**: Map-backed keyring with reverse index for efficient lookups
- **UnionDockerKeyring**: Combines multiple keyrings, returning all matching credentials
- **DockerConfigProvider**: Interface for providers that supply docker configurations
- **CachingDockerConfigProvider**: Wraps a provider with time-based caching
- **AuthConfig**: Docker registry authentication credentials (username, password, tokens)
- **TrackedAuthConfig**: AuthConfig with source tracking (secret or service account origin)
- **DockerConfig**: Map of registry URLs to DockerConfigEntry credentials

## Key Functions

- **NewDefaultDockerKeyring()**: Creates a keyring using the default .dockercfg provider
- **Lookup()**: Finds credentials matching an image URL with glob pattern support
- **Add()**: Adds credentials to the keyring with source tracking
- **URLsMatch()**: Matches image URLs against credential glob patterns
- **ReadDockerConfigFile()**: Reads credentials from .docker/config.json or .dockercfg

## Design Notes

- Index is reverse-sorted so more specific paths match first (e.g., gcr.io/project before gcr.io)
- Supports wildcard matching in registry hostnames (*.docker.io)
- Default registry (DockerHub) is matched for images without explicit registry
- Credentials are tracked with source information for KubeletEnsureSecretPulledImages feature
- Auth field in config is base64-encoded "username:password" format
