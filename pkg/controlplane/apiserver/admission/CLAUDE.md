# Package: admission

## Purpose
This package provides admission plugin initialization for the generic control plane API server. It sets up webhook-based admission and quota configuration for admission controllers.

## Key Types

- **Config**: Holds configuration needed for admission plugin initialization, including LoopbackClientConfig and ExternalInformers
- **PluginInitializer**: Implements admission.PluginInitializer to inject dependencies into admission plugins that need quota configuration or excluded resources

## Key Functions

- **Config.New()**: Creates admission plugin initializers for webhooks and Kubernetes-specific plugins. Sets up authentication info resolver for webhooks and quota configuration
- **NewPluginInitializer()**: Constructs a PluginInitializer with quota configuration and excluded admission resources
- **PluginInitializer.Initialize()**: Injects quota configuration and excluded resources into plugins that implement the corresponding interfaces (WantsQuotaConfiguration, WantsExcludedAdmissionResources)

## Design Notes

- Separates webhook admission initialization from Kubernetes-specific admission initialization
- Uses interface-based dependency injection pattern - plugins declare what they need via interfaces
- Integrates with the exclusion package to define resources that should be excluded from certain admission checks
