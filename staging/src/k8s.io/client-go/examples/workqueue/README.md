# Workqueue Example

This example demonstrates how to write a controller which follows the states
of watched resources.

It demonstrates how to:
 * combine the workqueue with a cache to a full controller
 * synchronize the controller on startup

The example is based on https://git.k8s.io/community/contributors/devel/controllers.md.

## Running

```
# if outside of the cluster
go run *.go -kubeconfig=/my/config -logtostderr=true
```
