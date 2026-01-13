# Package: secrets

## Purpose
This package provides functionality to extract Docker registry credentials from Kubernetes Secrets and build a DockerKeyring for image pulls. It supports both legacy .dockercfg and modern .docker/config.json secret formats.

## Key Functions

- **MakeDockerKeyring()**: Creates a DockerKeyring from secrets, unioned with a default keyring
- **secretsToTrackedDockerConfigs()**: Extracts docker configs from secrets with coordinate tracking

## Supported Secret Types

- **kubernetes.io/dockerconfigjson**: Modern format with .dockerconfigjson key
- **kubernetes.io/dockercfg**: Legacy format with .dockercfg key

## Design Notes

- Returns default keyring unchanged if no valid docker secrets found
- Tracks secret coordinates (UID, namespace, name) for each credential source
- Uses UnionDockerKeyring to combine secret-based and default credentials
- Secret credentials take precedence over default keyring (listed first in union)
- Credentials from all valid secrets are merged into a single BasicDockerKeyring
