# client-go Examples

This directory contains examples that cover various use cases and functionality
for client-go.

### Auth plugins

Client configuration is typically loaded from kubeconfig files containing server and credential configuration.
Several plugins for obtaining credentials from external sources are available, but are not loaded by default.
To enable these plugins in your program, import them in your main package.

You can load all auth plugins:
```go
import _ "k8s.io/client-go/plugin/pkg/client/auth"
```

Or you can load specific auth plugins:
```go
import _ "k8s.io/client-go/plugin/pkg/client/auth/azure"
import _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
import _ "k8s.io/client-go/plugin/pkg/client/auth/oidc"
```

### Configuration

- [**Authenticate in cluster**](./in-cluster-client-configuration): Configure a
  client while running inside the Kubernetes cluster.
- [**Authenticate out of cluster**](./out-of-cluster-client-configuration):
  Configure a client to access a Kubernetes cluster from outside.

### Basics

- [**Managing resources with API**](./create-update-delete-deployment): Create,
  get, update, delete a Deployment resource.

### Advanced Concepts

- [**Work queues**](./workqueue): Create a hotloop-free controller with the
  rate-limited workqueue and the [informer framework][informer].
- [**Custom Resource Definition (successor of TPR)**](https://git.k8s.io/apiextensions-apiserver/examples/client-go):
  Register a custom resource type with the API, create/update/query this custom
  type, and write a controller that drives the cluster state based on the changes to
  the custom resources.
- [**Leader election**](./leader-election): Demonstrates the use of the leader election package, which can be used to implement HA controllers.

[informer]: https://godoc.org/k8s.io/client-go/tools/cache#NewInformer

### Testing

- [**Fake Client**](./fake-client): Use a fake client in tests.
