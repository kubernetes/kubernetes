# Package: apiserver

## Purpose
This package implements the core control plane API server for Kubernetes. It provides configuration, setup, and lifecycle management for the kube-apiserver, including API aggregation, extensions, authentication, authorization, and peer proxy functionality.

## Key Types

- **Config / Extra**: Main configuration structures holding generic server config plus control-plane-specific settings (storage, service accounts, peer proxy, informers)
- **CompletedConfig**: Immutable, validated configuration ready for server creation
- **Server**: The control plane server instance wrapping GenericAPIServer with additional control plane components
- **RESTStorageProvider**: Interface for factories that create REST storage for API groups
- **APIServicePriority**: Defines group/version priority for API discovery ordering

## Key Functions

- **BuildGenericConfig()**: Constructs the generic apiserver config from options, setting up authentication, authorization, storage, OpenAPI, and informers
- **CreateConfig()**: Creates control-plane-specific config including proxy transport, admission plugins, and service account settings
- **CreateAggregatorConfig() / CreateAggregatorServer()**: Sets up the API aggregator layer that handles APIService registration and proxying to extension servers
- **CreateAPIExtensionsConfig()**: Configures the apiextensions-apiserver for CRD support
- **InstallAPIs()**: Installs REST storage providers for all enabled API groups
- **BuildPeerProxy() / CreatePeerEndpointLeaseReconciler()**: Sets up peer-to-peer proxying between API servers for version interoperability

## Design Notes

- Uses a delegation pattern where aggregator -> apiextensions -> kube-apiserver delegate to each other
- Implements the "completed config" pattern to ensure configuration is validated before use
- Registers multiple PostStartHooks for controllers: system namespaces, identity leases, cluster authentication trust, leader election
- Supports peer proxy feature for routing requests between API servers when one cannot serve a particular API version
- API groups have defined priorities that control their ordering in discovery responses
