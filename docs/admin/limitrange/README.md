<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
Limit Range
========================================
By default, pods run with unbounded CPU and memory limits.  This means that any pod in the
system will be able to consume as much CPU and memory on the node that executes the pod.

Users may want to impose restrictions on the amount of resource a single pod in the system may consume
for a variety of reasons.

For example:

1. Each node in the cluster has 2GB of memory.  The cluster operator does not want to accept pods
that require more than 2GB of memory since no node in the cluster can support the requirement.  To prevent a
pod from being permanently unscheduled to a node, the operator instead chooses to reject pods that exceed 2GB
of memory as part of admission control.
2. A cluster is shared by two communities in an organization that runs production and development workloads
respectively.  Production workloads may consume up to 8GB of memory, but development workloads may consume up
to 512MB of memory.  The cluster operator creates a separate namespace for each workload, and applies limits to
each namespace.
3. Users may create a pod which consumes resources just below the capacity of a machine.  The left over space
may be too small to be useful, but big enough for the waste to be costly over the entire cluster.  As a result,
the cluster operator may want to set limits that a pod must consume at least 20% of the memory and cpu of their
average node size in order to provide for more uniform scheduling and to limit waste.

This example demonstrates how limits can be applied to a Kubernetes namespace to control
min/max resource limits per pod.  In addition, this example demonstrates how you can
apply default resource limits to pods in the absence of an end-user specified value.

See [LimitRange design doc](../../design/admission_control_limit_range.md) for more information. For a detailed description of the Kubernetes resource model, see [Resources](../../../docs/user-guide/compute-resources.md)

Step 0: Prerequisites
-----------------------------------------
This example requires a running Kubernetes cluster.  See the [Getting Started guides](../../../docs/getting-started-guides/) for how to get started.

Change to the `<kubernetes>` directory if you're not already there.

Step 1: Create a namespace
-----------------------------------------
This example will work in a custom namespace to demonstrate the concepts involved.

Let's create a new namespace called limit-example:

```console
$ kubectl create -f docs/admin/limitrange/namespace.yaml
namespace "limit-example" created
$ kubectl get namespaces
NAME            LABELS    STATUS    AGE
default         <none>    Active    5m
limit-example   <none>    Active    53s
```

Step 2: Apply a limit to the namespace
-----------------------------------------
Let's create a simple limit in our namespace.

```console
$ kubectl create -f docs/admin/limitrange/limits.yaml --namespace=limit-example
limitrange "mylimits" created
```

Let's describe the limits that we have imposed in our namespace.

```console
$ kubectl describe limits mylimits --namespace=limit-example
Name:   mylimits
Namespace:  limit-example
Type        Resource      Min      Max      Request      Limit      Limit/Request
----        --------      ---      ---      -------      -----      -------------
Pod         cpu           200m     2        -            -          -
Pod         memory        6Mi      1Gi      -            -          -
Container   cpu           100m     2        200m         300m       -
Container   memory        3Mi      1Gi      100Mi        200Mi      -
```

In this scenario, we have said the following:

1. If a max constraint is specified for a resource (2 CPU and 1Gi memory in this case), then a limit
must be specified for that resource across all containers. Failure to specify a limit will result in
a validation error when attempting to create the pod. Note that a default value of limit is set by
*default* in file `limits.yaml` (300m CPU and 200Mi memory).
2. If a min constraint is specified for a resource (100m CPU and 3Mi memory in this case), then a
request must be specified for that resource across all containers. Failure to specify a request will
result in a validation error when attempting to create the pod. Note that a default value of request is
set by *defaultRequest* in file `limits.yaml` (200m CPU and 100Mi memory).
3. For any pod, the sum of all containers memory requests must be >= 6Mi and the sum of all containers
memory limits must be <= 1Gi; the sum of all containers CPU requests must be >= 200m and the sum of all
containers CPU limits must be <= 2.

Step 3: Enforcing limits at point of creation
-----------------------------------------
The limits enumerated in a namespace are only enforced when a pod is created or updated in
the cluster.  If you change the limits to a different value range, it does not affect pods that
were previously created in a namespace.

If a resource (cpu or memory) is being restricted by a limit, the user will get an error at time
of creation explaining why.

Let's first spin up a replication controller that creates a single container pod to demonstrate
how default values are applied to each pod.

```console
$ kubectl run nginx --image=nginx --replicas=1 --namespace=limit-example
replicationcontroller "nginx" created
$ kubectl get pods --namespace=limit-example
NAME          READY     STATUS    RESTARTS   AGE
nginx-aq0mf   1/1       Running   0          35s
$ kubectl get pods nginx-aq0mf --namespace=limit-example -o yaml | grep resources -C 8
```

```yaml
  resourceVersion: "127"
  selfLink: /api/v1/namespaces/limit-example/pods/nginx-aq0mf
  uid: 51be42a7-7156-11e5-9921-286ed488f785
spec:
  containers:
  - image: nginx
    imagePullPolicy: IfNotPresent
    name: nginx
    resources:
      limits:
        cpu: 300m
        memory: 200Mi
      requests:
        cpu: 200m
        memory: 100Mi
    terminationMessagePath: /dev/termination-log
    volumeMounts:
```

Note that our nginx container has picked up the namespace default cpu and memory resource *limits* and *requests*.

Let's create a pod that exceeds our allowed limits by having it have a container that requests 3 cpu cores.

```console
$ kubectl create -f docs/admin/limitrange/invalid-pod.yaml --namespace=limit-example
Error from server: error when creating "docs/admin/limitrange/invalid-pod.yaml": Pod "invalid-pod" is forbidden: [Maximum cpu usage per Pod is 2, but limit is 3., Maximum cpu usage per Container is 2, but limit is 3.]
```

Let's create a pod that falls within the allowed limit boundaries.

```console
$ kubectl create -f docs/admin/limitrange/valid-pod.yaml --namespace=limit-example
pod "valid-pod" created
$ kubectl get pods valid-pod --namespace=limit-example -o yaml | grep -C 6 resources
```

```yaml
 uid: 162a12aa-7157-11e5-9921-286ed488f785
spec:
  containers:
  - image: gcr.io/google_containers/serve_hostname
    imagePullPolicy: IfNotPresent
    name: kubernetes-serve-hostname
    resources:
      limits:
        cpu: "1"
        memory: 512Mi
      requests:
        cpu: "1"
        memory: 512Mi
```

Note that this pod specifies explicit resource *limits* and *requests* so it did not pick up the namespace
default values.

Note: The *limits* for CPU resource are not enforced in the default Kubernetes setup on the physical node
that runs the container unless the administrator deploys the kubelet with the folllowing flag:

```
$ kubelet --help
Usage of kubelet
....
  --cpu-cfs-quota[=false]: Enable CPU CFS quota enforcement for containers that specify CPU limits
$ kubelet --cpu-cfs-quota=true ...
```

Step 4: Cleanup
----------------------------
To remove the resources used by this example, you can just delete the limit-example namespace.

```console
$ kubectl delete namespace limit-example
namespace "limit-example" deleted
$ kubectl get namespaces
NAME      LABELS    STATUS    AGE
default   <none>    Active    20m
```

Summary
----------------------------
Cluster operators that want to restrict the amount of resources a single container or pod may consume
are able to define allowable ranges per Kubernetes namespace.  In the absence of any explicit assignments,
the Kubernetes system is able to apply default resource *limits* and *requests* if desired in order to
constrain the amount of resource a pod consumes on a node.






<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/limitrange/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
