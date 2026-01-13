# Package: main (sample-generic-controlplane)

## Purpose
This is the main entry point for the sample-generic-controlplane binary. It provides a kube-like control plane with CRDs, generic Kubernetes native APIs, and API aggregation, but without container-domain-specific APIs like pods and nodes.

## Key Functions

- **main()**: Entry point that creates and runs the server command using component-base CLI utilities

## Features

- CRD support via apiextensions-apiserver
- Generic Kubernetes native APIs (namespaces, RBAC, service accounts, etc.)
- API aggregation support
- JSON logging support
- Prometheus metrics for client-go and version

## Design Notes

- Uses the server package from samples/generic/server for actual server implementation
- Registers metrics and logging components via blank imports
- Designed as a reference implementation for building custom Kubernetes-style control planes
