## Purpose

This repository contains the definitions for the Container Runtime Interface (CRI).
CRI is a plugin interface which enables kubelet to use a wide variety of container runtimes,
without the need to recompile. CRI consists of a protocol buffers and gRPC API.
Read more about CRI API at [kubernetes docs](https://kubernetes.io/docs/concepts/architecture/cri/).

The repository [kubernetes/cri-api](https://github.com/kubernetes/cri-api) is a mirror of https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/cri-api.
Please do NOT file issues or submit PRs against the [kubernetes/cri-api](https://github.com/kubernetes/cri-api)
repository as it is readonly, all development is done in [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).

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

### Contibution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information. Please note that [kubernetes/cri-api](https://github.com/kubernetes/cri-api)
is a readonly mirror repository, all development is done at [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).
