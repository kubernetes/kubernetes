# Third Party Resources Example – Deprecated

**Note:** ThirdPartyResources are deprecated since 1.7. The successor is CustomResourceDefinition in the apiextensions.k8s.io API group.

This particular example demonstrates how to perform basic operations such as:

* How to register a new ThirdPartyResource (custom Resource type)
* How to create/get/list instances of your new Resource type (update/delete/etc work as well but are not demonstrated) 
* How to setup a controller on Resource handling create/update/delete events

## Running

```
# assumes you have a working kubeconfig, not required if operating in-cluster
go run *.go -kubeconfig=$HOME/.kube/config
```

## Use Cases

ThirdPartyResources can be used to implement custom Resource types for your Kubernetes cluster.
These act like most other Resources in Kubernetes, and may be `kubectl apply`'d, etc.

Some example use cases:

* Provisioning/Management of external datastores/databases (eg. CloudSQL/RDS instances)
* Higher level abstractions around Kubernetes primitives (eg. a single Resource to define an etcd cluster, backed by a Service and a ReplicationController) 

## Defining types

Each instance of your ThirdPartyResource has an attached Spec, which should be defined via a `struct{}` to provide data format validation.
In practice, this Spec is arbitrary key-value data that specifies the configuration/behavior of your Resource.

For example, if you were implementing a ThirdPartyResource for a Database, you might provide a DatabaseSpec like the following:

``` go
type DatabaseSpec struct {
	Databases []string `json:"databases"`
	Users     []User   `json:"users"`
	Version   string   `json:"version"`
}

type User struct {
	Name     string `json:"name"`
	Password string `json:"password"`
}
```