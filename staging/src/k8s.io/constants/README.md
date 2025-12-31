# constants

This module contains well-known constants used throughout Kubernetes.

It has zero dependencies, making it suitable for external consumers who need
Kubernetes constants without pulling in larger modules like `k8s.io/apimachinery`
or `k8s.io/api`.

## Packages

- `rfc/` - DNS naming limits from RFC 1123 and RFC 1035
- `limits/` - Kubernetes-specific size limits (labels, fields)
- `labels/` - Well-known label keys
- `annotations/` - Well-known annotation keys
- `taints/` - Well-known taint keys

## Usage

```go
import (
    "k8s.io/constants/rfc"
    "k8s.io/constants/labels"
)

// Use DNS constants
if len(name) > rfc.DNS1123SubdomainMaxLength {
    // ...
}

// Use well-known labels
zone := node.Labels[labels.LabelTopologyZone]
```

## Compatibility

Constants are re-exported from their original locations in `k8s.io/apimachinery`
and `k8s.io/api` for backwards compatibility.

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

You can reach the maintainers of this project at:

- [Slack](https://kubernetes.slack.com/messages/sig-architecture)
- [Mailing List](https://groups.google.com/g/kubernetes-sig-architecture)

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
