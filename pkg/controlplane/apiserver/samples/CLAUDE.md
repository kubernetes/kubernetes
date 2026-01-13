# Package: samples

## Purpose
This package contains sample kube-like generic control plane API servers. These samples demonstrate how to construct control planes that serve Kubernetes-style APIs without container-domain-specific resources.

## Contents

The package provides two sample implementations:
1. **generic**: A control plane with CRDs, generic Kube native APIs, and aggregation
2. **minimum**: A minimal control plane without CRDs (referenced in docs but may be in separate location)

## Design Notes

- These samples exist primarily to validate that generic control planes can be constructed using the Kubernetes API server libraries
- They may eventually be promoted as examples for third parties building custom control planes
- The samples lack container-specific APIs (pods, nodes, deployments, etc.) but retain core APIs like namespaces, RBAC, and service accounts
