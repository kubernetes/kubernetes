# Package: kubeapiserver

## Purpose
This package holds code common to the kube-apiserver and federation-apiserver that is not part of the generic API server. It provides storage factory configuration and default settings specific to Kubernetes API servers.

## Key Types

- **StorageFactoryConfig**: Configuration for creating storage factory with encoding overrides
- **completedStorageFactoryConfig**: Wrapper for completed storage configuration

## Key Functions

- **NewStorageFactoryConfig()**: Creates a new storage factory config with resource overrides
- **NewStorageFactoryConfigEffectiveVersion()**: Creates config respecting emulation version
- **Complete()**: Completes storage config with etcd options
- **New()**: Creates DefaultStorageFactory from completed configuration

## Key Variables

- **SpecialDefaultResourcePrefixes**: Maps resources to custom etcd key prefixes
- **DefaultWatchCacheSizes()**: Defines which resources disable watch cache (events)

## Design Notes

- Configures cohabitating resources (e.g., apps/deployments and extensions/deployments share storage)
- Supports etcd server overrides per resource via --etcd-servers-overrides flag
- Handles storage version encoding for specific resources during version transitions
- Uses legacyscheme.Codecs as the default serializer
