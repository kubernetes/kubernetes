# Package: admission

## Purpose
This package provides Kubernetes-specific admission plugin initialization for the kube-apiserver. It implements the plugin initializer interface to inject dependencies into admission plugins during startup.

## Key Types

- **Config**: Configuration holder for admission plugin initialization
- **PluginInitializer**: Kubernetes-specific admission plugin initializer

## Key Functions

- **Config.New()**: Returns a slice of plugin initializers for admission setup
- **NewPluginInitializer()**: Creates a new Kubernetes admission plugin initializer
- **Initialize()**: Initializes admission plugins with Kubernetes-specific dependencies

## Design Notes

- Currently a minimal implementation - the Initialize() method is empty
- Designed to be extended with Kubernetes-specific initialization (e.g., WantsToRun with stopCh)
- Works alongside generic admission initializers from apiserver package
- The plugin initializer pattern allows plugins to receive dependencies via interface checks
