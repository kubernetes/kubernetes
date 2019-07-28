## Purpose

This repository contains functions to be consumed by various Kubernetes and
out-of-tree CSI components like external provisioner to facilitate migration of
code from Kubernetes In-tree plugin code to CSI plugin repositories.

Consumers of this repository can make use of functions like `TranslateToCSI` and
`TranslateToInTree` functions to translate PV sources.

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community
page](http://kubernetes.io/community/).

You can reach the maintainers of this repository at:

- Slack: #sig-storage (on https://kubernetes.slack.com -- get an
  invite at slack.kubernetes.io)
- Mailing List:
  https://groups.google.com/forum/#!forum/kubernetes-sig-storage

### Code of Conduct

Participation in the Kubernetes community is governed by the [Kubernetes
Code of Conduct](code-of-conduct.md).

### Contibution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
