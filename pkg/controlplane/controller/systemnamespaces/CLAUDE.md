# Package: systemnamespaces

## Purpose
This controller ensures that required system namespaces exist in the cluster. It creates namespaces like kube-system, kube-public, and default if they are missing.

## Key Types

- **Controller**: Periodically checks and creates system namespaces

## Key Functions

- **NewController()**: Creates a controller with a list of system namespaces to ensure exist
- **Run()**: Starts the controller, syncing every 1 minute
- **sync()**: Iterates through system namespaces and creates any that are missing
- **createNamespaceIfNeeded()**: Creates a namespace if it doesn't already exist

## Default System Namespaces

Typically includes:
- kube-system: For Kubernetes system components
- kube-public: For publicly readable data
- default: The default namespace for user workloads

## Design Notes

- Uses namespace informer/lister for efficient existence checks
- Ignores AlreadyExists errors to handle race conditions in HA clusters
- Runs every 1 minute to handle namespace deletion and re-creation
- Simple and robust - just ensures namespaces exist, no other management
