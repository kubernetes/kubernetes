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
[here](http://releases.k8s.io/release-1.0/docs/admin/resourcequota/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Resource Quota
========================================
This example demonstrates how [resource quota](../../admin/admission-controllers.md#resourcequota) and
[limitsranger](../../admin/admission-controllers.md#limitranger) can be applied to a Kubernetes namespace.
See [ResourceQuota design doc](../../design/admission_control_resource_quota.md) for more information.

This example assumes you have a functional Kubernetes setup.

Step 1: Create a namespace
-----------------------------------------
This example will work in a custom namespace to demonstrate the concepts involved.

Let's create a new namespace called quota-example:

```console
$ kubectl create -f docs/admin/resourcequota/namespace.yaml
namespace "quota-example" created
$ kubectl get namespaces
NAME            LABELS    STATUS    AGE
default         <none>    Active    2m
quota-example   <none>    Active    39s
```

Step 2: Apply a quota to the namespace
-----------------------------------------
By default, a pod will run with unbounded CPU and memory requests/limits.  This means that any pod in the
system will be able to consume as much CPU and memory on the node that executes the pod.

Users may want to restrict how much of the cluster resources a given namespace may consume
across all of its pods in order to manage cluster usage.  To do this, a user applies a quota to
a namespace.  A quota lets the user set hard limits on the total amount of node resources (cpu, memory)
and API resources (pods, services, etc.) that a namespace may consume. In term of resources, Kubernetes
checks the total resource *requests*, not resource *limits* of all containers/pods in the namespace.

Let's create a simple quota in our namespace:

```console
$ kubectl create -f docs/admin/resourcequota/quota.yaml --namespace=quota-example
resourcequota "quota" created
```

Once your quota is applied to a namespace, the system will restrict any creation of content
in the namespace until the quota usage has been calculated.  This should happen quickly.

You can describe your current quota usage to see what resources are being consumed in your
namespace.

```console
$ kubectl describe quota quota --namespace=quota-example
Name:			quota
Namespace:		quota-example
Resource		Used	Hard
--------		----	----
cpu			0	20
memory			0	1Gi
persistentvolumeclaims	0	10
pods			0	10
replicationcontrollers	0	20
resourcequotas		1	1
secrets			1	10
services		0	5
```

Step 3: Applying default resource requests and limits
-----------------------------------------
Pod authors rarely specify resource requests and limits for their pods.

Since we applied a quota to our project, let's see what happens when an end-user creates a pod that has unbounded
cpu and memory by creating an nginx container.

To demonstrate, lets create a replication controller that runs nginx:

```console
$ kubectl run nginx --image=nginx --replicas=1 --namespace=quota-example
replicationcontroller "nginx" created
```

Now let's look at the pods that were created.

```console
$ kubectl get pods --namespace=quota-example
NAME      READY     STATUS    RESTARTS   AGE
```

What happened?  I have no pods!  Let's describe the replication controller to get a view of what is happening.

```console
kubectl describe rc nginx --namespace=quota-example
Name:		nginx
Namespace:	quota-example
Image(s):	nginx
Selector:	run=nginx
Labels:		run=nginx
Replicas:	0 current / 1 desired
Pods Status:	0 Running / 0 Waiting / 0 Succeeded / 0 Failed
No volumes.
Events:
  FirstSeen	LastSeen	Count	From				SubobjectPath	Reason		Message
  42s		11s		3	{replication-controller }			FailedCreate	Error creating: Pod "nginx-" is forbidden: Must make a non-zero request for memory since it is tracked by quota.
```

The Kubernetes API server is rejecting the replication controllers requests to create a pod because our pods
do not specify any memory usage *request*.

So let's set some default values for the amount of cpu and memory a pod can consume:

```console
$ kubectl create -f docs/admin/resourcequota/limits.yaml --namespace=quota-example
limitrange "limits" created
$ kubectl describe limits limits --namespace=quota-example
Name:		limits
Namespace:	quota-example
Type		Resource	Min	Max	Request	Limit	Limit/Request
----		--------	---	---	-------	-----	-------------
Container	memory		-	-	256Mi	512Mi	-
Container	cpu		-	-	100m	200m	-
```

Now any time a pod is created in this namespace, if it has not specified any resource request/limit, the default
amount of cpu and memory per container will be applied, and the request will be used as part of admission control.

Now that we have applied default resource *request* for our namespace, our replication controller should be able to
create its pods.

```console
$ kubectl get pods --namespace=quota-example
NAME          READY     STATUS    RESTARTS   AGE
nginx-fca65   1/1       Running   0          1m
```

And if we print out our quota usage in the namespace:

```console
$ kubectl describe quota quota --namespace=quota-example
Name:			quota
Namespace:		quota-example
Resource		Used	Hard
--------		----	----
cpu			100m	20
memory			256Mi	1Gi
persistentvolumeclaims	0	10
pods			1	10
replicationcontrollers	1	20
resourcequotas		1	1
secrets			1	10
services		0	5
```

You can now see the pod that was created is consuming explicit amounts of resources (specified by resource *request*),
and the usage is being tracked by the Kubernetes system properly.

Summary
----------------------------
Actions that consume node resources for cpu and memory can be subject to hard quota limits defined
by the namespace quota. The resource consumption is measured by resource *request* in pod specification.

Any action that consumes those resources can be tweaked, or can pick up namespace level defaults to
meet your end goal.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/resourcequota/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
