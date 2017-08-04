# External Repository Staging Area

This directory is the staging area for packages that have been split to their
own repository. The content here will be periodically published to respective
top-level k8s.io repositories.

Repositories currently staged here:

- [`k8s.io/apiextensions-apiserver`](https://github.com/kubernetes/apiextensions-apiserver)
- [`k8s.io/api`](https://github.com/kubernetes/api)
- [`k8s.io/apimachinery`](https://github.com/kubernetes/apimachinery)
- [`k8s.io/apiserver`](https://github.com/kubernetes/apiserver)
- [`k8s.io/client-go`](https://github.com/kubernetes/client-go)
- [`k8s.io/kube-aggregator`](https://github.com/kubernetes/kube-aggregator)
- [`k8s.io/kube-gen`](https://github.com/kubernetes/kube-gen) (about to be published)
- [`k8s.io/metrics`](https://github.com/kubernetes/metrics)
- [`k8s.io/sample-apiserver`](https://github.com/kubernetes/sample-apiserver)

The code in the staging/ directory is authoritative, i.e. the only copy of the
code. You can directly modify such code.

## Using staged repositories from Kubernetes code

Kubernetes code uses the repositories in this directory via symlinks in the
`vendor/k8s.io` directory into this staging area.  For example, when
Kubernetes code imports a package from the `k8s.io/client-go` repository, that
import is resolved to `staging/src/k8s.io/client-go` relative to the project
root:

```go
// pkg/example/some_code.go
package example

import (
  "k8s.io/client-go/dynamic" // resolves to staging/src/k8s.io/client-go/dynamic
)
```

Once the change-over to external repositories is complete, these repositories
will actually be vendored from `k8s.io/<package-name>`.
