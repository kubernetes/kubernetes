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

## Cluster setup

The container runtime must support CDI. The latest cri-o releases contain
support, containerd 1.6.x does not. To bring up a kind cluster with containerd
built from their main branch, use:

- [Bash version requirement](https://github.com/kubernetes/community/blob/master/contributors/devel/development.md#bash-version-requirement)

- Build node image

```bash
$ test/e2e/dra/kind-build-image.sh dra/node:latest
```

- Bring up a kind cluster

```bash
$ kind create cluster --config test/e2e/dra/kind.yaml --image dra/node:latest
```


## Run tests

- Build ginkgo

> NB: If you are using go workspace you must disable it `GOWORK=off make gingko`

```bash
$ make gingko
```

- Run e2e tests for the `Dynamic Resource Allocation` feature:

```bash
$ KUBECONFIG=~/.kube/config _output/bin/ginkgo -p -v -focus=Feature:DynamicResourceAllocation ./test/e2e
```
