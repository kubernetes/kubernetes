# Feature Documentation

This directory contains documentation for new Kubernetes features.

## Features

- **feat1.md**: `kubectl get pods --stale` - Show pods that haven't received traffic in X time
- **feat2.md**: `kubectl get pods --idle` - Show pods that haven't been accessed (exec, port-forward, etc.) in X time

## Quick Test Guide

After cloning this repository, see [TESTING.md](./TESTING.md) for detailed instructions.

### Quick Start (TL;DR)

```bash
# 1. Build kubectl (or just run 'make' to build everything)
make all WHAT=cmd/kubectl

# 2. Create kind cluster
kind create cluster --name test-cluster

# 3. Get kubeconfig
kind get kubeconfig --name test-cluster > /tmp/test-kubeconfig.yaml

# 4. Test the features
KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl get pods --stale=1h

KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl get pods --idle=30m
```

### Automated Testing

Run the automated test script:

```bash
./_docs/test-features.sh
```

This will build kubectl, create a test cluster, create test pods, and run both feature tests.

## Requirements

- Go 1.21+
- kind
- Docker or compatible container runtime
- make

