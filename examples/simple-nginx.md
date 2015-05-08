## Running your first containers in Kubernetes

Ok, you've run one of the [getting started guides](../docs/getting-started-guides/) and you have
successfully turned up a Kubernetes cluster.  Now what?  This guide will help you get oriented
to Kubernetes and running your first containers on the cluster.

### Running a container (simple version)

From this point onwards, it is assumed that ```kubectl``` is on your path from one of the getting started guides.

The `kubectl` line below spins up two containers running
[Nginx](http://nginx.org/en/) running on port 80:

```bash
kubectl run-container my-nginx --image=nginx --replicas=2 --port=80
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
On some platforms (for example Google Compute Engine) the kubectl command can integrate with your cloud provider to add a public IP address for the pods,
to do this run:

```bash
kubectl expose rc my-nginx --port=80 --create-external-load-balancer
```

This should print the service that has been created, and map an external IP address to the service.

### Next: Configuration files
Most people will eventually want to use declarative configuration files for creating/modifying their applications.  A [simplified introduction](simple-yaml.md)
is given in a different document.
