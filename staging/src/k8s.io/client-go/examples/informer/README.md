# Informer Example

Informers provide a high-level API for creating custom controllers for Kubernetes resources.

This particular example demonstrates:

* How to write an Informer against a core resource type.
* How to handle add, update and delete events.

The [thirdparty-resources example](../third-party-resources) demonstrates how to use informers against a thirdparty resource.

## Running

To run the example outside the Kubernetes cluster you need to supply the path to a Kubernetes config file.

```sh
go run main.go -kubeconfig=$HOME/.kube/config -logtostderr
```

By default `glog` logs to files.
Use the `-logtostderr` command line argument so that you can see the output on the console.

## Running Inside a Kubernetes Cluster

You can also run the example inside a Kubernetes cluster.
In this case it will use [in Cluster configuration](../in-cluster/),
and you don't need to supply `-kubeconfig` or `-master` command line flags.

## Use Cases

* Building controllers that coordinate other resources.
  Most controllers in [k8s.io/kubernetes/pkg/controller](https://godoc.org/k8s.io/kubernetes/pkg/controller) use informers.
* Capturing resource events for logging to external systems
  (e.g. monitor non-"Normal" events and publish metrics to a time series database)
