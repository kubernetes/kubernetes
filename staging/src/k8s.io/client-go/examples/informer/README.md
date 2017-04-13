# Informer Example

Informers provide a high-level API for creating custom controllers for Kubernetes resources.

This particular example demonstrates:

* How to write an Informer against a core resource type and handle to create/update/delete events

## Running

``` go
# assumes you have a working kubeconfig, not required if operating in-cluster
go run *.go -kubeconfig=$HOME/.kube/config
```

## Use Cases

* Capturing resource events for logging to external systems (e.g. monitor non-"Normal" events and publish metrics to a time series database)
* Creating lifecycle controllers for ThirdPartyResources (eg. coordinate create/update/delete of an external datastore represented via a ThirdPartyResource type)
