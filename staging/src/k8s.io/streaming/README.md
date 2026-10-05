> ⚠️ **This is an automatically published [staged repository](https://git.k8s.io/kubernetes/staging#external-repository-staging-area) for Kubernetes**.   
> Contributions, including issues and pull requests, should be made to the main Kubernetes repository: [https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).  
> This repository is read-only for importing, and not used for direct contributions.  
> See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

# streaming

This repository contains the Kubernetes HTTP streaming transport primitives used for:

- generic stream upgrade negotiation
- SPDY stream connections and round-tripping
- WebSocket channel streaming helpers

The goal of this module is to provide a dedicated import target for transport utilities shared by CRI streaming, client-go, apiserver, and kubectl.

## Migration notes

- The legacy package path `k8s.io/apimachinery/pkg/util/httpstream` was intentionally removed as part of this extraction.
- Consumers must migrate imports to:
  - `k8s.io/streaming/pkg/httpstream`
  - `k8s.io/streaming/pkg/httpstream/spdy`
  - `k8s.io/streaming/pkg/httpstream/wsstream`
- This extraction does not provide compatibility shims at the old apimachinery path.

## Community, discussion, contribution, and support

streaming is maintained as part of [SIG API Machinery](https://github.com/kubernetes/community/tree/master/sig-api-machinery) and [SIG Node](https://github.com/kubernetes/community/tree/master/sig-node) areas.

You can reach maintainers of this project at:

- Slack: [#sig-node](https://kubernetes.slack.com/messages/sig-node)
- Mailing List: [kubernetes-sig-node](https://groups.google.com/forum/#!forum/kubernetes-sig-node)

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
