# client-go Examples

This directory contains examples that cover various use cases and functionality
for client-go.

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
- [**Third-party resources (deprecated)**](./third-party-resources-deprecated):
  Register a third-party resource type with the API, create/update/query this third-party
  type, and write a controller that drives the cluster state based on the changes to
  the third-party resources.
- [**Custom Resource Definition (successor of TPR)**](https://git.k8s.io/apiextensions-apiserver/examples/client-go):
  Register a custom resource type with the API, create/update/query this custom
  type, and write a controller that drives the cluster state based on the changes to
  the custom resources.

[informer]: https://godoc.org/k8s.io/client-go/tools/cache#NewInformer
