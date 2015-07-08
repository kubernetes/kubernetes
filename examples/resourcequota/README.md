Resource Quota
========================================
This example demonstrates how resource quota and limits can be applied to a Kubernetes namespace.

This example assumes you have a functional Kubernetes setup.

Step 1: Create a namespace
-----------------------------------------
This example will work in a custom namespace to demonstrate the concepts involved.

Let's create a new namespace called quota-example:

```shell
$ kubectl create -f namespace.yaml
$ kubectl get namespaces
NAME            LABELS             STATUS
default         <none>             Active
quota-example   <none>             Active
```

Step 2: Apply a quota to the namespace
-----------------------------------------
By default, a pod will run with unbounded CPU and memory limits.  This means that any pod in the
system will be able to consume as much CPU and memory on the node that executes the pod.

Users may want to restrict how much of the cluster resources a given namespace may consume
across all of its pods in order to manage cluster usage.  To do this, a user applies a quota to
a namespace.  A quota lets the user set hard limits on the total amount of node resources (cpu, memory)
and API resources (pods, services, etc.) that a namespace may consume.

Let's create a simple quota in our namespace:

```shell
$ kubectl create -f quota.yaml --namespace=quota-example
```

Once your quota is applied to a namespace, the system will restrict any creation of content
in the namespace until the quota usage has been calculated.  This should happen quickly.

You can describe your current quota usage to see what resources are being consumed in your
namespace.

```
$ kubectl describe quota quota --namespace=quota-example
Name:                   quota
Namespace:              quota-example
Resource                Used    Hard
--------                ----    ----
cpu                     0       20
memory                  0       1Gi
persistentvolumeclaims  0       10
pods                    0       10
replicationcontrollers  0       20
resourcequotas          1       1
secrets                 1       10
services                0       5
```

Step 3: Applying default resource limits
-----------------------------------------
Pod authors rarely specify resource limits for their pods.

Since we applied a quota to our project, let's see what happens when an end-user creates a pod that has unbounded
cpu and memory by creating an nginx container.

To demonstrate, lets create a replication controller that runs nginx:

```shell
$ kubectl run nginx --image=nginx --replicas=1 --namespace=quota-example
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR    REPLICAS
nginx        nginx          nginx      run=nginx   1
```

Now let's look at the pods that were created.

```shell
$ kubectl get pods --namespace=quota-example
NAME      READY     STATUS    RESTARTS   AGE
```

What happened?  I have no pods!  Let's describe the replication controller to get a view of what is happening.

```shell
kubectl describe rc nginx --namespace=quota-example
Name:   nginx
Image(s): nginx
Selector: run=nginx
Labels:   run=nginx
Replicas: 0 current / 1 desired
Pods Status:  0 Running / 0 Waiting / 0 Succeeded / 0 Failed
Events:
  FirstSeen       LastSeen      Count From        SubobjectPath Reason    Message
  Mon, 01 Jun 2015 22:49:31 -0400 Mon, 01 Jun 2015 22:52:22 -0400 7 {replication-controller }     failedCreate  Error creating: Pod "nginx-" is forbidden: Limited to 1Gi memory, but pod has no specified memory limit
```

The Kubernetes API server is rejecting the replication controllers requests to create a pod because our pods
do not specify any memory usage.

So let's set some default limits for the amount of cpu and memory a pod can consume:

```shell
$ kubectl create -f limits.yaml --namespace=quota-example
limitranges/limits
$ kubectl describe limits limits --namespace=quota-example
Name:           limits
Namespace:      quota-example
Type            Resource        Min     Max     Default
----            --------        ---     ---     ---
Container       memory          -       -       512Mi
Container       cpu             -       -       100m
```

Now any time a pod is created in this namespace, if it has not specified any resource limits, the default
amount of cpu and memory per container will be applied as part of admission control.

Now that we have applied default limits for our namespace, our replication controller should be able to
create its pods.

```shell
$ kubectl get pods --namespace=quota-example
NAME          READY     STATUS    RESTARTS   AGE
nginx-t9cap   1/1       Running   0          49s
```

And if we print out our quota usage in the namespace:

```shell
kubectl describe quota quota --namespace=quota-example
Name:                   quota
Namespace:              default
Resource                Used            Hard
--------                ----            ----
cpu                     100m            20
memory                  536870912       1Gi
persistentvolumeclaims  0               10
pods                    1               10
replicationcontrollers  1               20
resourcequotas          1               1
secrets                 1               10
services                0               5
```

You can now see the pod that was created is consuming explicit amounts of resources, and the usage is being
tracked by the Kubernetes system properly.

Summary
----------------------------
Actions that consume node resources for cpu and memory can be subject to hard quota limits defined
by the namespace quota.

Any action that consumes those resources can be tweaked, or can pick up namespace level defaults to
meet your end goal.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/resourcequota/README.md?pixel)]()
