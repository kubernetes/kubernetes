# cri-client

Container Runtime Interface client implementation

## Purpose

This repository contains the client source code for the Container Runtime Interface (CRI).
CRI is a plugin interface which enables kubelet to use a wide variety of container runtimes,
without the need to recompile. CRI consists of a protocol buffers and gRPC API.
Read more about CRI at [kubernetes docs](https://kubernetes.io/docs/concepts/architecture/cri/).

The repository [kubernetes/cri-client](https://github.com/kubernetes/cri-client) is a mirror of https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/cri-client.
Please do **not** file issues or submit PRs against the [kubernetes/cri-client](https://github.com/kubernetes/cri-client)
repository as it is readonly, all development is done in [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).

The [`cri-api` staging repository](https://github.com/kubernetes/cri-api) is
defined in [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)
and is **only** intended to be used for kubelet to container runtime
interactions, or for node-level troubleshooting using a tool such as `crictl`.
The `cri-api` wraps the [Protobuf protocol definition](https://github.com/kubernetes/cri-api/blob/63929b3/pkg/apis/runtime/v1/api.proto)
with extra features which can be used directly from other clients as well.
It is **not** a common purpose container runtime API for general use, and is
intended to be Kubernetes-centric. We try to avoid it, but there may be logic
within a container runtime that optimizes for the order or specific parameters
of call(s) that the kubelet makes.

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community
page](http://kubernetes.io/community/).

You can reach the maintainers of this repository at:

- Slack: #sig-node (on https://kubernetes.slack.com -- get an
  invite at [slack.kubernetes.io](https://slack.kubernetes.io))
- Mailing List:
  https://groups.google.com/forum/#!forum/kubernetes-sig-node

### Code of Conduct

Participation in the Kubernetes community is governed by the [Kubernetes
Code of Conduct](code-of-conduct.md).

### Contribution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information. Please note that [kubernetes/cri-api](https://github.com/kubernetes/cri-api)
is a readonly mirror repository, all development is done at [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).
