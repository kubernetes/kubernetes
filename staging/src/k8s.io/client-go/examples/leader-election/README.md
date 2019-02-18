# Leader Election Example

This example demonstrates how to write a leader election in Kubernetes.

## Running

The example needs a unique `POD_NAME` environment variable, as well as a common `POD_NAMESPACE` variable.

Run the following three commands in separate terminals.

```bash
# if outside of the cluster
# first terminal 
$ POD_NAME=pod-1 POD_NAMESPACE=default go run *.go -kubeconfig=/my/config -logtostderr=true

# second terminal 
$ POD_NAME=pod-2 POD_NAMESPACE=default go run *.go -kubeconfig=/my/config -logtostderr=true

# third terminal
$ POD_NAME=pod-3 POD_NAMESPACE=default go run *.go -kubeconfig=/my/config -logtostderr=true
```

Now kill the existing leader. Via leader election one of these two pods is selected as the new leader, and you should see the leader failover to a different pod.