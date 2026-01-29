# Pod Logs

This example demonstrates how to retrieve and print logs from all pods in all namespaces.

## Running the example

Make sure you have a Kubernetes cluster and `kubectl` is configured.

Compile this example on your workstation:

```bash
cd pod-logs
go build -o app .
```

Now, run this application on your workstation with your local kubeconfig file:

```bash
./app
# or specify a kubeconfig file with flag
./app -kubeconfig=$HOME/.kube/config
```

The example will list each pod's name, namespace, IP, and node, then print the last 5 lines of logs for each container within those pods.
