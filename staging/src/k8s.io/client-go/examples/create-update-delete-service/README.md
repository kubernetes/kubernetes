# Create, Update & Delete Service

This example program demonstrates the fundamental operations for managing on
[Service][1] resources, such as `Create`, `List`, `Update` and `Delete`.

You can adopt the source code from this example to write programs that manage
other types of resources through the Kubernetes API.

## Running this example

Make sure you have a Kubernetes cluster and `kubectl` is configured:

    kubectl get nodes

Compile this example on your workstation:

```
cd create-update-delete-service
go build -o ./app
```

Now, run this application on your workstation with your local kubeconfig file:

```
./app
# or specify a kubeconfig file with flag
./app -kubeconfig=$HOME/.kube/config
```

Running this command will execute the following operations on your cluster:

1. **Create service:** This will create a  Service. Verify with
   `kubectl get services`.
2. **Update Service:** This will update the Service resource created in
   previous step by changing the type of service
   `kubectl describe service demo`.
3. **List Services:** This will retrieve Services in the `default`
   namespace and print their names and type.
4. **Delete Service:** This will delete the Service object. Verify with `kubectl get services`.

Each step is separated by an interactive prompt. You must hit the
<kbd>Enter</kbd> key to proceed to the next step. You can use these prompts as
a break to take time to run `kubectl` and inspect the result of the operations
executed.

You should see an output like the following:

```
Creating Service...
Created service "demo-service".
-> Press Enter key to continue.

Updating Service...
Updated service...
-> Press Enter key to continue.

Listing services in namespace "default":
 * demo-services (NodePort)
-> Press Enter key to continue.

Deleting service...
Deleted service.
```

## Cleanup

Successfully running this program will clean the created artifacts. If you
terminate the program without completing, you can clean up the created
service with:

    kubectl delete service demo-service

## Troubleshooting

If you are getting the following error, make sure Kubernetes version of your
cluster is v1.6 or above in `kubectl version`:

    panic: the server could not find the requested resource

[1]: https://kubernetes.io/docs/user-guide/service/
