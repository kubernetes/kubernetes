# dra-test-driver

This driver implements the controller and a resource kubelet plugin for dynamic
resource allocation. This is done in a single binary to minimize the amount of
boilerplate code. "Real" drivers could also implement both in different
binaries.

## Usage

The driver could get deployed as a Deployment for the controller, with leader
election. A DaemonSet could get used for the kubelet plugin. The controller can
also run as a Kubernetes client outside of a cluster. The same works for the
kubelet plugin when using port forwarding. This is how it is used during
testing.

Valid parameters are key/value string pairs stored in a ConfigMap.
Those get copied into the ResourceClaimStatus with "user_" and "admin_" as
prefix, depending on whether they came from the ResourceClaim or ResourceClass.
They get stored in the `ResourceHandle` field as JSON map by the controller.
The kubelet plugin then sets these attributes as environment variables in each
container that uses the resource.

Resource availability is configurable and can simulate different scenarios:

- Network-attached resources, available on all nodes where the node driver runs, or
  host-local resources, available only on the node whether they were allocated.
- Shared or unshared allocations.
- Unlimited or limited resources. The limit is a simple number of allocations
  per cluster or node.

While the functionality itself is very limited, the code strives to showcase
best practices and supports metrics, leader election, and the same logging
options as Kubernetes.

## Design

The binary itself is a Cobra command with two operations, `controller` and
`kubelet-plugin`. Logging is done with [contextual
logging](https://github.com/kubernetes/enhancements/tree/master/keps/sig-instrumentation/3077-contextual-logging).

The `k8s.io/dynamic-resource-allocation/controller` package implements the
interaction with ResourceClaims. It is generic and relies on an interface to
implement the actual driver logic. Long-term that part could be split out into
a reusable utility package.

The `k8s.io/dynamic-resource-allocation/kubelet-plugin` package implements the
interaction with kubelet, again relying only on the interface defined for the
kubelet<->dynamic resource allocation plugin interaction.

`app` is the driver itself with a very simple implementation of the interfaces.

## Deployment

### `local-up-cluster.sh`

To try out the feature, build Kubernetes, then in one console run:
```console
RUNTIME_CONFIG="resource.k8s.io/v1alpha3" FEATURE_GATES=DynamicResourceAllocation=true ALLOW_PRIVILEGED=1 ./hack/local-up-cluster.sh -O
```

In another:
```console
go run ./test/e2e/dra/test-driver --feature-gates ContextualLogging=true -v=5 controller
```

In yet another:
```console
sudo mkdir -p /var/run/cdi && sudo chmod a+rwx /var/run/cdi /var/lib/kubelet/plugins_registry
go run ./test/e2e/dra/test-driver --feature-gates ContextualLogging=true -v=5 kubelet-plugin --node-name=127.0.0.1
```

And finally:
```console
$ kubectl create -f test/e2e/dra/test-driver/deploy/example/resourceclass.yaml
resourceclass/example created
$ kubectl create -f test/e2e/dra/test-driver/deploy/example/pod-inline.yaml
configmap/pause-claim-parameters created
pod/pause created

$ kubectl get resourceclaims
NAME             CLASSNAME   ALLOCATIONMODE         STATE                AGE
pause-resource   example     WaitForFirstConsumer   allocated,reserved   19s

$ kubectl get pods
NAME    READY   STATUS    RESTARTS   AGE
pause   1/1     Running   0          23s
```

There are also examples for other scenarios (multiple pods, multiple claims).

### multi-node cluster

At this point there are no container images that contain the test driver and
therefore it cannot be deployed on "normal" clusters.

## Prior art

Some of this code was derived from the
[external-resizer](https://github.com/kubernetes-csi/external-resizer/). `controller`
corresponds to the [controller
logic](https://github.com/kubernetes-csi/external-resizer/blob/master/pkg/controller/controller.go),
which in turn is similar to the
[sig-storage-lib-external-provisioner](https://github.com/kubernetes-sigs/sig-storage-lib-external-provisioner).
