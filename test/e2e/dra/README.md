# Overview

The tests in this directory cover dynamic resource allocation support in
Kubernetes. They do not test the correct behavior of arbitrary dynamic resource
allocation drivers.

If such a driver is needed, then the in-tree test/e2e/dra/test-driver is used,
with a slight twist: instead of deploying that driver directly in the cluster,
the necessary sockets for interaction with kubelet (registration and dynamic
resource allocation) get proxied into the e2e.test binary. This reuses the work
done for CSI mock testing. The advantage is that no separate images are needed
for the test driver and that the e2e test has full control over all gRPC calls,
in case that it needs that for operations like error injection or checking
calls.

## Cluster setup preparation

The container runtime must support CDI. CRI-O supports CDI starting from release 1.23,
Containerd supports CDI starting from release 1.7. To bring up a Kind cluster with Containerd,
two things are needed:
- [build binaries from Kubernetes source code tree](https://github.com/kubernetes/community/blob/master/contributors/devel/development.md#building-kubernetes)
- [Kind](https://github.com/kubernetes-sigs/kind)

> NB: Kind switched to use worker-node base image with Containerd 1.7 by default starting from
release 0.20, build kind from latest main branch sources or use Kind release binary 0.20 or later.

### Build kind node image

After building Kubernetes, in Kubernetes source code tree build new node image:
```bash
$ kind build node-image --image dra/node:latest $(pwd)
```

## Bring up a Kind cluster
```bash
$ kind create cluster --config test/e2e/dra/kind.yaml --image dra/node:latest
```


## Run tests

- Build ginkgo

```bash
$ make ginkgo
```

- Run e2e tests for the `Dynamic Resource Allocation` feature:

```bash
$ KUBECONFIG=~/.kube/config _output/bin/ginkgo -p -v -focus=Feature:DynamicResourceAllocation ./test/e2e
```
