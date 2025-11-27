# Testing kubectl --stale and --idle Features

This guide explains how to test the new `kubectl get pods --stale` and `--idle` features after cloning this repository.

## Prerequisites

- Go 1.21+ installed
- [kind](https://kind.sigs.k8s.io/) installed
- Docker or compatible container runtime
- `make` installed

## Quick Start

### 1. Build kubectl

Build the kubectl binary with the new features:

```bash
# Option 1: Build just kubectl (faster, recommended)
make all WHAT=cmd/kubectl

# Option 2: Build everything (slower, but works too)
make
```

This will create the binary at `_output/local/bin/darwin/arm64/kubectl` (or `linux/amd64` on Linux, or your platform's equivalent).

### 2. Create or Use a Kind Cluster

Create a new kind cluster:

```bash
kind create cluster --name test-cluster
```

Or use an existing cluster.

### 3. Get Kubeconfig

Save the kubeconfig for easy access:

```bash
kind get kubeconfig --name test-cluster > /tmp/test-kubeconfig.yaml
```

### 4. Test the Features

#### Test --stale flag

```bash
# Use the locally built kubectl
KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl get pods --stale=1h

# Or test across all namespaces
KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl get pods -A --stale=1h
```

#### Test --idle flag

```bash
# Show pods idle for more than 30 minutes (shows IDLE-SINCE column)
KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl get pods --idle=30m

# Default to 30m if no duration specified
KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl get pods --idle
```

### 5. Create Test Pods (Optional)

To see the features in action, create some test pods:

```bash
KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl run test-pod --image=nginx --restart=Never

# Wait a bit, then check idle pods
sleep 60
KUBECONFIG=/tmp/test-kubeconfig.yaml \
  ./_output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl get pods --idle=1m
```

## Automated Test Script

For convenience, use the provided test script:

```bash
./_docs/test-features.sh
```

This script will:
1. Build kubectl
2. Create a kind cluster (if one doesn't exist)
3. Create test pods
4. Run both `--stale` and `--idle` tests
5. Show the results

## Troubleshooting

### kubectl binary not found

Make sure you've built kubectl:
```bash
make all WHAT=cmd/kubectl
```

Check the binary location:
```bash
ls -la _output/local/bin/$(go env GOOS)/$(go env GOARCH)/kubectl
```

### Kind cluster issues

List existing clusters:
```bash
kind get clusters
```

Delete and recreate if needed:
```bash
kind delete cluster --name test-cluster
kind create cluster --name test-cluster
```

### Features not working

- Ensure you're using the **locally built** kubectl binary, not the system kubectl
- The system kubectl won't have these features until they're merged upstream
- Check that you're pointing to the correct kubeconfig

## Testing Kubelet Features (Advanced)

If you've also modified kubelet and want to test the `/pods/idle` endpoint:

1. Build a custom kind node image:
```bash
kind build node-image --image kindest/node:custom .
```

2. Create cluster with custom image:
```bash
kind create cluster --name test-cluster --image kindest/node:custom
```

3. Test the kubelet endpoint:
```bash
docker exec test-cluster-control-plane curl -s http://localhost:10255/pods/idle
```

Note: The kubelet endpoint requires kubelet changes to be built into the node image.

