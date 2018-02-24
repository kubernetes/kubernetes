# Controller References Manager

This example program demonstrates how controllers can use
`BaseControllerRefManager` to adopt and release resources. See
[ControllerRef proposal][1] for more details.

## Running this example

Make sure you have a Kubernetes cluster and `kubectl` is configured:

```
kubectl get nodes
```
Run this application on your workstation with your local kubeconfig file:

```
go run main.go -kubeconfig=$HOME/.kube/config
```

Running this command will execute the following operations on your cluster:

1. **Create Deployment:** This will create a Deployment without
`OwnerReferences`. Verify with `kubectl get deployment demo-deployment -o yaml`.
2. **Adopt Deployment:** This will adopt the Deployment resource created in
previous step by `MyController`. Verify with
`kubectl get deployment demo-deployment -o yaml`.
4. **Delete Deployment:** This will delete the Deployment object.
Verify with `kubectl get deployments`.

Each step is separated by an interactive prompt. You must hit the
<kbd>Return</kbd> key to proceed to the next step. You can use these prompts as
a break to take time to  run `kubectl` and inspect the result of the operations
executed.

## Cleanup

Successfully running this program will clean the created artifacts. If you
terminate the program without completing, you can clean up the created
deployment with:

    kubectl delete deploy demo-deployment

## Troubleshooting

If you are getting the following error, make sure Kubernetes version of your
cluster is v1.8 or above in `kubectl version`:

    panic: the server could not find the requested resource

[1]: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md
