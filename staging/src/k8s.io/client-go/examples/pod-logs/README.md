# Pod Logs

This example lists pods and prints recent container logs with `client-go`.

## Running this example

Make sure you have a Kubernetes cluster and `kubectl` is configured:

    kubectl get nodes

Compile this example on your workstation:

```
cd pod-logs
go build -o ./app
```

Run this application with your local kubeconfig file:

```
./app
# or specify a kubeconfig file with flag
./app -kubeconfig=$HOME/.kube/config
```

By default the example reads the last 10 lines from each container in each pod
across all namespaces. You can narrow the result:

```
./app -namespace=default
./app -namespace=default -pod=nginx
./app -namespace=default -pod=nginx -container=web
./app -namespace=default -pod=nginx -container=web -tail=25
```

To stream new log lines as they are written:

```
./app -namespace=default -pod=nginx -container=web -follow
```
