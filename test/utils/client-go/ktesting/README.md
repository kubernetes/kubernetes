# ktesting

ktesting is a set of packages with the same name which are meant to simplify
writing tests.

- k8s.io/klog/v2/ktesting [![Go Reference](https://pkg.go.dev/badge/k8s.io/klog/v2/ktesting.svg)](https://pkg.go.dev/k8s.io/klog/v2/ktesting): per-test structured logging
- k8s.io/kubernetes/test/utils/ktesting [![Go Reference](https://pkg.go.dev/badge/k8s.io/kubernetes/test/utils/ktesting.svg)](https://pkg.go.dev/k8s.io/kubernetes/test/utils/ktesting): API that enables writing code which can be used in tests based on `go test` (unit tests and integration tests in Kubernetes) and Ginkgo suites (E2E in Kubernetes). Automatically adds klog command line flags and reconfigures Gomega to log Kubernetes objects as YAML.
- k8s.io/kubernetes/test/utils/client-go/ktesting [![Go Reference](https://pkg.go.dev/badge/k8s.io/kubernetes/test/utils/client-go/ktesting.svg)](https://pkg.go.dev/k8s.io/kubernetes/test/utils/client-go/ktesting): Adds support for client-go to the API (client instance, test namespace, REST configuration), similar to test/e2e/framework.Framework.

They build on top of each other such that the more advanced packages offer the
same API as the simpler packages and just add more functionality.

Providing them in different modules avoids forcing dependencies on consumers
for functionality that they don't need. Consumers are free to choose the
package which meets their needs and can later switch to a more advanced one
just by changing the import statement.
