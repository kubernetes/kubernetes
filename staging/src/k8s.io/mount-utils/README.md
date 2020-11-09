## Purpose

This repository defines an interface to mounting filesystems to be consumed by 
various Kubernetes and out-of-tree CSI components. 

Consumers of this repository can make use of functions like 'Mount' to mount 
source to target as fstype with given options, 'Unmount' to unmount a target.
Other useful functions include 'List' all mounted file systems and find all
mount references to a path using 'GetMountRefs'

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
