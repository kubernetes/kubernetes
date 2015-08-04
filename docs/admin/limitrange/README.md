<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/admin/limitrange/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

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

Change to the `<kubernetes>/examples/limitrange` directory if you're not already there.

Step 1: Create a namespace
-----------------------------------------
This example will work in a custom namespace to demonstrate the concepts involved.

Let's create a new namespace called limit-example:

```console
$ kubectl create -f docs/admin/limitrange/namespace.yaml
namespaces/limit-example
$ kubectl get namespaces
NAME            LABELS             STATUS
default         <none>             Active
limit-example   <none>             Active
```

Step 2: Apply a limit to the namespace
-----------------------------------------
Let's create a simple limit in our namespace.

```console
$ kubectl create -f docs/admin/limitrange/limits.yaml --namespace=limit-example
limitranges/mylimits
```

Let's describe the limits that we have imposed in our namespace.

```console
$ kubectl describe limits mylimits --namespace=limit-example
Name:   mylimits
Type      Resource  Min  Max Default
----      --------  ---  --- ---
Pod       memory    6Mi  1Gi -
Pod       cpu       250m   2 -
Container memory    6Mi  1Gi 100Mi
Container cpu       250m   2 250m
```

In this scenario, we have said the following:

1. The total memory usage of a pod across all of its container must fall between 6Mi and 1Gi.
2. The total cpu usage of a pod across all of its containers must fall between 250m and 2 cores.
3. A container in a pod may consume between 6Mi and 1Gi of memory.  If the container does not
specify an explicit resource limit, each container in a pod will get 100Mi of memory.
4. A container in a pod may consume between 250m and 2 cores of cpu.  If the container does
not specify an explicit resource limit, each container in a pod will get 250m of cpu.

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
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR    REPLICAS
nginx        nginx          nginx      run=nginx   1
$ kubectl get pods --namespace=limit-example
POD           IP           CONTAINER(S)   IMAGE(S)   HOST          LABELS      STATUS    CREATED          MESSAGE
nginx-ykj4j   10.246.1.3                             10.245.1.3/   run=nginx   Running   About a minute
                           nginx          nginx                                Running   54 seconds
$ kubectl get pods nginx-ykj4j --namespace=limit-example -o yaml | grep resources -C 5
```

```yaml
  containers:
  - capabilities: {}
    image: nginx
    imagePullPolicy: IfNotPresent
    name: nginx
    resources:
      limits:
        cpu: 250m
        memory: 100Mi
    terminationMessagePath: /dev/termination-log
    volumeMounts:
```

Note that our nginx container has picked up the namespace default cpu and memory resource limits.

Let's create a pod that exceeds our allowed limits by having it have a container that requests 3 cpu cores.

```console
$ kubectl create -f docs/admin/limitrange/invalid-pod.yaml --namespace=limit-example
Error from server: Pod "invalid-pod" is forbidden: Maximum CPU usage per pod is 2, but requested 3
```

Let's create a pod that falls within the allowed limit boundaries.

```console
$ kubectl create -f docs/admin/limitrange/valid-pod.yaml --namespace=limit-example
pods/valid-pod
$ kubectl get pods valid-pod --namespace=limit-example -o yaml | grep -C 5 resources
```

```yaml
  containers:
  - capabilities: {}
    image: gcr.io/google_containers/serve_hostname
    imagePullPolicy: IfNotPresent
    name: nginx
    resources:
      limits:
        cpu: "1"
        memory: 512Mi
    securityContext:
      capabilities: {}
```

Note that this pod specifies explicit resource limits so it did not pick up the namespace default values.

Step 4: Cleanup
----------------------------
To remove the resources used by this example, you can just delete the limit-example namespace.

```console
$ kubectl delete namespace limit-example
namespaces/limit-example
$ kubectl get namespaces
NAME      LABELS    STATUS
default   <none>    Active
```

Summary
----------------------------
Cluster operators that want to restrict the amount of resources a single container or pod may consume
are able to define allowable ranges per Kubernetes namespace.  In the absence of any hard limits,
the Kubernetes system is able to apply default resource limits if desired in order to constrain the
amount of resource a pod consumes on a node.




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/limitrange/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
