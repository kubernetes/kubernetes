# Package: format

## Purpose
The `format` package provides consistent formatting utilities for pods, producing human-readable strings for logging and debugging.

## Key Functions

- **Pod**: Returns a formatted string for a pod in the format `name_namespace(uid)`. Returns `<nil>` for nil pods.
- **PodDesc**: Returns a formatted string from individual pod name, namespace, and UID components.

## Output Format

The output format is: `podName_podNamespace(podUID)`

Example: `nginx_default(abc123-def456)`

## Design Notes

- Uses underscore as delimiter because it's not allowed in pod names (DNS subdomain format) but is allowed in container names.
- Provides consistent formatting for pod identification across logs and error messages.
- Handles nil pod safely by returning `<nil>`.
