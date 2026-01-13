# Package: defaults

## Purpose
Provides a default KubeSchedulerConfiguration for use in tests. This package offers a convenient way to get a fully-defaulted scheduler configuration without needing to manually construct one.

## Key Functions

- **GetDefaultKubeSchedulerConfiguration()**: Returns a complete KubeSchedulerConfiguration with all default values applied. It creates a config using the `v1` scheme, applies defaults via `configv1.SetDefaults_KubeSchedulerConfiguration`, and converts it back to the internal `config.KubeSchedulerConfiguration` type.

## Usage Notes
- This is a testing utility package - use it in tests that need a valid scheduler configuration
- The returned configuration includes all default plugins, profiles, and settings
- Uses the scheme from `configv1.SchemeGroupVersion` for proper defaulting
