# constants

This module contains well-known constants used throughout Kubernetes.

It has zero dependencies, making it suitable for external consumers who need
Kubernetes constants without pulling in larger modules like `k8s.io/apimachinery`
or `k8s.io/api`.

## Packages

- `rfc/` — length constants for Kubernetes name-like fields (historically named after DNS 1123/1035)
- `limits/` — Kubernetes-specific size limits (label value/key, field manager)
- `labels/` — well-known label keys
- `annotations/` — well-known annotation keys
- `taints/` — well-known node taints

Identifiers in `labels/`, `annotations/`, and `taints/` intentionally omit the
package-name prefix (`labels.Hostname`, not `labels.LabelHostname`).

## Usage

```go
import (
    "k8s.io/constants/rfc"
    "k8s.io/constants/labels"
)

// Length constant
if len(name) > rfc.DNS1123SubdomainMaxLength {
    // ...
}

// Well-known label
zone := node.Labels[labels.TopologyZone]
```

## Compatibility

All constants that previously lived in `k8s.io/apimachinery` and `k8s.io/api/core/v1`
are re-exported from their original locations and retain their legacy names, so
existing code does not need to change.

## Migrating from apimachinery/api to k8s.io/constants

New code, and code that wants to minimise its dependency footprint, should import
from `k8s.io/constants/*` directly. The legacy import paths continue to work; this
module is simply the lighter-weight source.

### Length and size constants

Previously sourced from `k8s.io/apimachinery/pkg/util/validation` or
`k8s.io/apimachinery/pkg/apis/meta/v1/validation`:

```go
// Before
import "k8s.io/apimachinery/pkg/util/validation"
_ = validation.DNS1123SubdomainMaxLength
_ = validation.DNS1123LabelMaxLength
_ = validation.DNS1035LabelMaxLength
_ = validation.LabelValueMaxLength

import metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
_ = metav1validation.FieldManagerMaxLength

// After
import (
    "k8s.io/constants/rfc"
    "k8s.io/constants/limits"
)
_ = rfc.DNS1123SubdomainMaxLength
_ = rfc.DNS1123LabelMaxLength
_ = rfc.DNS1035LabelMaxLength
_ = limits.LabelValueMaxLength
_ = limits.FieldManagerMaxLength
```

### Well-known labels

Previously sourced from `k8s.io/api/core/v1`:

```go
// Before
import corev1 "k8s.io/api/core/v1"
zone := node.Labels[corev1.LabelTopologyZone]

// After
import "k8s.io/constants/labels"
zone := node.Labels[labels.TopologyZone]
```

The `Label` prefix is dropped. Legacy identifiers (`LabelTopologyZone`,
`LabelHostname`, …) remain available in `k8s.io/api/core/v1` as re-exports.

### Well-known annotations

Previously sourced from `k8s.io/api/core/v1` or `k8s.io/kubectl/pkg/cmd/apply`:

```go
// Before
import corev1 "k8s.io/api/core/v1"
v := obj.Annotations[corev1.AnnotationTopologyMode]

// After
import "k8s.io/constants/annotations"
v := obj.Annotations[annotations.TopologyMode]
```

The `Annotation` prefix and `AnnotationKey` suffix are dropped. Deprecated
annotations carry a `Deprecated` prefix and a godoc `Deprecated:` line pointing
to the replacement field or constant.

### Well-known taints

Previously sourced from `k8s.io/api/core/v1`:

```go
// Before
import corev1 "k8s.io/api/core/v1"
if taint.Key == corev1.TaintNodeNotReady { /* ... */ }

// After
import "k8s.io/constants/taints"
if taint.Key == taints.NodeNotReady { /* ... */ }
```

The `Taint` prefix is dropped. Legacy identifiers (`TaintNodeNotReady`, …)
remain available in `k8s.io/api/core/v1` as re-exports.

### Dependency footprint

Importing `k8s.io/constants/*` brings in no transitive dependencies. Importing
`k8s.io/api/core/v1` or `k8s.io/apimachinery/pkg/util/validation` pulls in the
whole API type tree or the full validation package respectively.

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

You can reach the maintainers of this project at:

- [Slack](https://kubernetes.slack.com/messages/sig-architecture)
- [Mailing List](https://groups.google.com/g/kubernetes-sig-architecture)

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
