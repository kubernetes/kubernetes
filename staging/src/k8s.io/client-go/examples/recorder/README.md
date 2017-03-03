# Recorder/Events Example

This example demonstrates how to send and receive events. By default it will
print all events associated to the pod `kube-apiserver-master`. Further, it
will check every 10 seconds if this pod exists at all, and if it finds the pod,
it emits a `PodDiscovered` event, which is will be bound to that pod.

The result is, that you will see all events associated with that pod,
printed to `stdout`, including the `PodDiscovered` event.

It demonstrates how to:
 * use an event recorder for sending events to the cluster
 * watch for events in general
 * watch for events belonging to a specific object

## Running

```
# if outside of the cluster
go run *.go -kubeconfig=/my/config -logtostderr=true
```
