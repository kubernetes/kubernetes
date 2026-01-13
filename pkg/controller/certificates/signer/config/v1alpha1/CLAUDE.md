# Package: signer/config/v1alpha1

## Purpose
Provides defaulting functions for the v1alpha1 CSR signing controller configuration.

## Key Functions
- `RecommendedDefaultCSRSigningControllerConfiguration(obj)`: Applies recommended defaults to configuration

## Default Values
- `ClusterSigningDuration`: 365 days (1 year) if not specified

## Design Notes
- Uses metav1.Duration for time duration configuration
- Only defaults ClusterSigningDuration; cert/key file paths must be explicitly configured
- Called during configuration loading to ensure sensible defaults
- Intentionally not registered as scheme defaulter to allow embedding packages to control defaulting
