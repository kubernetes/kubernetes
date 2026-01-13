# Package: features

## Purpose
This package defines all Kubernetes feature gates and manages their registration with the default feature gate instance. Feature gates allow enabling/disabling experimental or alpha/beta functionality at runtime via the --feature-gates flag.

## Key Types

- **Feature constants**: Named feature gate identifiers (e.g., AnyVolumeDataSource, CoordinatedLeaderElection)
- **clientAdapter**: Adapts component-base feature gate to client-go's Gate and Registry interfaces

## Key Functions

- **init()**: Registers all default Kubernetes feature gates with DefaultMutableFeatureGate
- **clientAdapter.Add()**: Converts client-go feature specs to component-base format
- **clientAdapter.AddVersioned()**: Adds versioned feature gates for emulation version support
- **clientAdapter.Enabled()**: Checks if a client-go feature is enabled
- **clientAdapter.Set()**: Sets a feature gate value (used in testing)

## Feature Gate Lifecycle

- **Alpha**: Disabled by default, may be removed
- **Beta**: Enabled by default, may graduate to GA
- **GA**: Always enabled, cannot be disabled
- **Deprecated**: Scheduled for removal

## Design Notes

- Features are listed alphabetically with owner and KEP references
- Uses versioned feature specs to support emulation versions
- Integrates features from multiple sources: apiserver, apiextensions, client-go, controller-manager
- Feature dependencies can be declared for automatic enablement
- The clientAdapter allows client-go features to be controlled via the same --feature-gates flag
