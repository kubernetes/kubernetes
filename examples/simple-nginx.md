<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Running your first containers in Kubernetes

Ok, you've run one of the [getting started guides](../docs/getting-started-guides/) and you have
successfully turned up a Kubernetes cluster.  Now what?  This guide will help you get oriented
to Kubernetes and running your first containers on the cluster.

### Running a container (simple version)

From this point onwards, it is assumed that `kubectl` is on your path from one of the getting started guides.

The [`kubectl run`](../docs/user-guide/kubectl/kubectl_run.md) line below will create two [nginx](https://registry.hub.docker.com/_/nginx/) [pods](../docs/user-guide/pods.md) listening on port 80. It will also create a [replication controller](../docs/user-guide/replication-controller.md) named `my-nginx` to ensure that there are always two pods running.

```bash
kubectl run my-nginx --image=nginx --replicas=2 --port=80
```

Once the pods are created, you can list them to see what is up and running:

```bash
kubectl get pods
```

You can also see the replication controller that was created:

```bash
kubectl get rc
```

To stop the two replicated containers, stop the replication controller:

```bash
kubectl stop rc my-nginx
```

### Exposing your pods to the internet.

On some platforms (for example Google Compute Engine) the kubectl command can integrate with your cloud provider to add a [public IP address](../docs/user-guide/services.md#publishing-services---service-types) for the pods,
to do this run:

```bash
kubectl expose rc my-nginx --port=80 --type=LoadBalancer
```

This should print the service that has been created, and map an external IP address to the service. Where to find this external IP address will depend on the environment you run in.  For instance, for Google Compute Engine the external IP address is listed as part of the newly created service and can be retrieved by running

```bash
kubectl get services
```

In order to access your nginx landing page, you also have to make sure that traffic from external IPs is allowed. Do this by opening a firewall to allow traffic on port 80.

### Next: Configuration files

Most people will eventually want to use declarative configuration files for creating/modifying their applications.  A [simplified introduction](../docs/user-guide/simple-yaml.md)
is given in a different document.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/simple-nginx.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
