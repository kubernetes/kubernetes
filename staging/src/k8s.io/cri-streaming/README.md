> ⚠️ **This is an automatically published [staged repository](https://git.k8s.io/kubernetes/staging#external-repository-staging-area) for Kubernetes**.   
> Contributions, including issues and pull requests, should be made to the main Kubernetes repository: [https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).  
> This repository is read-only for importing, and not used for direct contributions.  
> See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

# cri-streaming

This repository contains the Kubernetes CRI streaming server implementation used for:

- Exec
- Attach
- PortForward

The goal of this module is to provide a dedicated, runtime-focused import target for CRI streaming functionality without requiring consumers to depend on the full `k8s.io/kubelet` module surface.

## Migration notes

- The legacy package path `k8s.io/kubelet/pkg/cri/streaming` has moved to `k8s.io/cri-streaming/pkg/streaming`.
- Shared transport dependencies now come from `k8s.io/streaming/pkg/httpstream` and subpackages.
- This extraction does not provide compatibility shims at the old kubelet/apimachinery paths.

## Community, discussion, contribution, and support

cri-streaming is planned as a sub-project of [SIG Node](https://github.com/kubernetes/community/tree/master/sig-node).

You can reach maintainers of this project at:

- Slack: [#sig-node](https://kubernetes.slack.com/messages/sig-node)
- Mailing List: [kubernetes-sig-node](https://groups.google.com/forum/#!forum/kubernetes-sig-node)

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
