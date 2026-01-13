# Package: install

## Purpose
Installs the apiserverinternal API group into the Kubernetes API encoding/decoding machinery, making it available for use across the API server.

## Key Functions
- `Install(scheme *runtime.Scheme)`: Registers the apiserverinternal API group and its v1alpha1 version to the provided scheme, setting version priority for serialization.
- `init()`: Automatically installs the API group to the legacy scheme on package import.

## Design Notes
- Uses the standard Kubernetes pattern for API group installation
- Registers both the internal version and v1alpha1 external version
- Sets v1alpha1 as the preferred version for serialization
