<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Replication Controller

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Replication Controller](#replication-controller)
  - [What is a _replication controller_?](#what-is-a-replication-controller)
  - [Running an example Replication Controller](#running-an-example-replication-controller)
  - [Writing a Replication Controller Spec](#writing-a-replication-controller-spec)
    - [Pod Template](#pod-template)
    - [Labels on the Replication Controller](#labels-on-the-replication-controller)
    - [Pod Selector](#pod-selector)
    - [Multiple Replicas](#multiple-replicas)
  - [Working with Replication Controllers](#working-with-replication-controllers)
    - [Deleting a Replication Controller and its Pods](#deleting-a-replication-controller-and-its-pods)
    - [Deleting just a Replication Controller](#deleting-just-a-replication-controller)
    - [Isolating pods from a Replication Controller](#isolating-pods-from-a-replication-controller)
  - [Common usage patterns](#common-usage-patterns)
    - [Rescheduling](#rescheduling)
    - [Scaling](#scaling)
    - [Rolling updates](#rolling-updates)
    - [Multiple release tracks](#multiple-release-tracks)
    - [Using Replication Controllers with Services](#using-replication-controllers-with-services)
  - [Writing programs for Replication](#writing-programs-for-replication)
  - [Responsibilities of the replication controller](#responsibilities-of-the-replication-controller)
  - [API Object](#api-object)
  - [Alternatives to Replication Controller](#alternatives-to-replication-controller)
    - [Bare Pods](#bare-pods)
    - [Job](#job)
    - [DaemonSet](#daemonset)

<!-- END MUNGE: GENERATED_TOC -->

## What is a _replication controller_?

A _replication controller_ ensures that a specified number of pod "replicas" are running at any one
time. In other words, a replication controller makes sure that a pod or homogeneous set of pods are
always up and available.
If there are too many pods, it will kill some. If there are too few, the
replication controller will start more. Unlike manually created pods, the pods maintained by a
replication controller are automatically replaced if they fail, get deleted, or are terminated.
For example, your pods get re-created on a node after disruptive maintenance such as a kernel upgrade.
For this reason, we recommend that you use a replication controller even if your application requires
only a single pod. You can think of a replication controller as something similar to a process supervisor,
but rather then individual processes on a single node, the replication controller supervises multiple pods
across multiple nodes.

Replication Controller is often abbreviated to "rc" or "rcs" in discussion, and as a shortcut in
kubectl commands.

A simple case is to create 1 Replication Controller object in order to reliably run one instance of
a Pod indefinitely.  A more complex use case is to run several identical replicas of a replicated
service, such as web servers.

## Running an example Replication Controller

Here is an example Replication Controller config.  It runs 3 copies of the nginx web server.
<!-- BEGIN MUNGE: EXAMPLE replication.yaml -->

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: nginx
spec:
  replicas: 3
  selector:
    app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
```

[Download example](replication.yaml?raw=true)
<!-- END MUNGE: EXAMPLE replication.yaml -->

Run the example job by downloading the example file and then running this command:

```console

$ kubectl create -f ./replication.yaml
replicationcontrollers/nginx

```

Check on the status of the replication controller using this command:

```console

$ kubectl describe replicationcontrollers/nginx
Name:		nginx
Namespace:	default
Image(s):	nginx
Selector:	app=nginx
Labels:		app=nginx
Replicas:	3 current / 3 desired
Pods Status:	0 Running / 3 Waiting / 0 Succeeded / 0 Failed
Events:
  FirstSeen				LastSeen			Count	From
SubobjectPath	Reason			Message
  Thu, 24 Sep 2015 10:38:20 -0700	Thu, 24 Sep 2015 10:38:20 -0700	1
{replication-controller }			SuccessfulCreate	Created pod: nginx-qrm3m
  Thu, 24 Sep 2015 10:38:20 -0700	Thu, 24 Sep 2015 10:38:20 -0700	1
{replication-controller }			SuccessfulCreate	Created pod: nginx-3ntk0
  Thu, 24 Sep 2015 10:38:20 -0700	Thu, 24 Sep 2015 10:38:20 -0700	1
{replication-controller }			SuccessfulCreate	Created pod: nginx-4ok8v

```

Here, 3 pods have been made, but none are running yet, perhaps because the image is being pulled.
A little later, the same command may show:

```
Pods Status:	3 Running / 0 Waiting / 0 Succeeded / 0 Failed

```

To list all the pods that belong to the rc in a machine readable form, you can use a command like this:

```console

$ pods=$(kubectl get pods --selector=app=nginx --output=jsonpath={.items..metadata.name})
echo $pods
nginx-3ntk0 nginx-4ok8v nginx-qrm3m

```

Here, the selector is the same as the selector for the replication controller (seen in the
`kubectl describe` output, and in a different form in `replication.yaml`.  The `--output=jsonpath` option
specifies an expression that just gets the name from each pod in the returned list.


## Writing a Replication Controller Spec

As with all other Kubernetes config, a Job needs `apiVersion`, `kind`, and `metadata` fields.  For
general information about working with config files, see [here](simple-yaml.md),
[here](configuring-containers.md), and [here](working-with-resources.md).

A Replication Controller also needs a [`.spec` section](../devel/api-conventions.md#spec-and-status).

### Pod Template

The `.spec.template` is the only required field of the `.spec`.

The `.spec.template` is a [pod template](replication-controller.md#pod-template).  It has exactly
the same schema as a [pod](pods.md), except it is nested and does not have an `apiVersion` or
`kind`.

In addition to required fields for a Pod, a pod template in a job must specify appropriate
labels (see [pod selector](#pod-selector) and an appropriate restart policy.

Only a [`RestartPolicy`](pod-states.md) equal to `Always` is allowed, which is the default
if not specified.

For local container restarts, replication controllers delegate to an agent on the node,
for example the [Kubelet](../admin/kubelet.md) or Docker.

### Labels on the Replication Controller

The replication controller can itself have labels (`.metadata.labels`).  Typically, you
would set these the same as the `.spec.template.metadata.labels`; if `.metadata.labels` is not specified
then it is defaulted to  `.spec.template.metadata.labels`.  However, they are allowed to be
different, and the `.metadata.labels` do not affec the behavior of the replication controller.

### Pod Selector

The `.spec.selector` field is a [label selector](labels.md#label-selectors).  A replication
controller manages all the pods with labels which match the selector.  It does not distinguish
between pods which it created or deleted versus pods which some other person or process created or
deleted.  This allows the replication controller to be replaced without affecting the running pods.

If specified, the `.spec.template.metadata.labels` must be equal to the `.spec.selector`, or it will
be rejected by the API.  If `.spec.selector` is unspecified, it will be defaulted to
`.spec.template.metadata.labels`.

Also you should not normally create any pods whose labels match this selector, either directly, via
another ReplicationController or via another controller such as Job.  Otherwise, the
ReplicationController will think that those pods were created by it.  Kubernetes will not stop you
from doing this.

If you do end up with multiple controllers that have overlapping selectors, you
will have to manage the deletion yourself (see [below](#updating-a-replication-controller)).

### Multiple Replicas

You can specify how many pods should run concurrently by setting `.spec.replicas` to the number
of pods you would like to have running concurrently.  The number running at any time may be higher
or lower, such as if the replicas was just increased or decreased, or if a pod is gracefully
shutdown, and a replacement starts early.

If you do not specify `.spec.replicas`, then it defaults to 1.

## Working with Replication Controllers

### Deleting a Replication Controller and its Pods

To delete a replication controller and all its pods, use [`kubectl
delete`](kubectl/kubectl_delete.md).  Kubectl will scale the replication controller to zero and wait
for it to delete each pod before deleting the replication controller itself.  If this kubectl
command is interrupted, it can be restarted.

When using the REST API or go client library, you need to do the steps explicitly (scale replicas to
0, wait for pod deletions, then delete the replication controller).

### Deleting just a Replication Controller

You can delete a replication controller without affecting any of its pods.

Using kubectl, specify the `--cascade=false` option to [`kubectl delete`](kubectl/kubectl_delete.md).

When using the REST API or go client library, simply delete the replication controller object.

Once the original is deleted, you can create a new replication controller to replace it.  As long
as the old and new `.spec.selector` are the same, then the new one will adopt the old pods.
However, it will not make any effort to make existing pods match a new, different pod template.
To update pods to a new spec in a controlled way, use a [rolling update](#rolling-updates).

### Isolating pods from a Replication Controller

Pods may be removed from a replication controller's target set by changing their labels. This technique may be used to remove pods from service for debugging, data recovery, etc. Pods that are removed in this way will be replaced automatically (assuming that the number of replicas is not also changed).

## Common usage patterns

### Rescheduling

As mentioned above, whether you have 1 pod you want to keep running, or 1000, a replication controller will ensure that the specified number of pods exists, even in the event of node failure or pod termination (e.g., due to an action by another control agent).

### Scaling

The replication controller makes it easy to scale the number of replicas up or down, either manually or by an auto-scaling control agent, by simply updating the `replicas` field.

### Rolling updates

The replication controller is designed to facilitate rolling updates to a service by replacing pods one-by-one.

As explained in [#1353](http://issue.k8s.io/1353), the recommended approach is to create a new replication controller with 1 replica, scale the new (+1) and old (-1) controllers one by one, and then delete the old controller after it reaches 0 replicas. This predictably updates the set of pods regardless of unexpected failures.

Ideally, the rolling update controller would take application readiness into account, and would ensure that a sufficient number of pods were productively serving at any given time.

The two replication controllers would need to create pods with at least one differentiating label, such as the image tag of the primary container of the pod, since it is typically image updates that motivate rolling updates.

Rolling update is implemented in the client tool
[kubectl](kubectl/kubectl_rolling-update.md)

### Multiple release tracks

In addition to running multiple releases of an application while a rolling update is in progress, it's common to run multiple releases for an extended period of time, or even continuously, using multiple release tracks. The tracks would be differentiated by labels.

For instance, a service might target all pods with `tier in (frontend), environment in (prod)`.  Now say you have 10 replicated pods that make up this tier.  But you want to be able to 'canary' a new version of this component.  You could set up a replication controller with `replicas` set to 9 for the bulk of the replicas, with labels `tier=frontend, environment=prod, track=stable`, and another replication controller with `replicas` set to 1 for the canary, with labels `tier=frontend, environment=prod, track=canary`.  Now the service is covering both the canary and non-canary pods.  But you can mess with the replication controllers separately to test things out, monitor the results, etc.

### Using Replication Controllers with Services

Multiple replication controllers can sit behind a single service, so that, for example, some traffic
goes to the old version, and some goes to the new version.

A replication controller will never terminate on its own, but it isn't expected to be as long-lived as services. Services may be composed of pods controlled by multiple replication controllers, and it is expected that many replication controllers may be created and destroyed over the lifetime of a service (for instance, to perform an update of pods that run the service). Both services themselves and their clients should remain oblivious to the replication controllers that maintain the pods of the services.

## Writing programs for Replication

Pods created by a replication controller are intended to be fungible and semantically identical, though their configurations may become heterogeneous over time. This is an obvious fit for replicated stateless servers, but replication controllers can also be used to maintain availability of master-elected, sharded, and worker-pool applications. Such applications should use dynamic work assignment mechanisms, such as the [etcd lock module](https://coreos.com/docs/distributed-configuration/etcd-modules/) or [RabbitMQ work queues](https://www.rabbitmq.com/tutorials/tutorial-two-python.html), as opposed to static/one-time customization of the configuration of each pod, which is considered an anti-pattern. Any pod customization performed, such as vertical auto-sizing of resources (e.g., cpu or memory), should be performed by another online controller process, not unlike the replication controller itself.

## Responsibilities of the replication controller

The replication controller simply ensures that the desired number of pods matches its label selector and are operational. Currently, only terminated pods are excluded from its count. In the future, [readiness](http://issue.k8s.io/620) and other information available from the system may be taken into account, we may add more controls over the replacement policy, and we plan to emit events that could be used by external clients to implement arbitrarily sophisticated replacement and/or scale-down policies.

The replication controller is forever constrained to this narrow responsibility. It itself will not perform readiness nor liveness probes. Rather than performing auto-scaling, it is intended to be controlled by an external auto-scaler (as discussed in [#492](http://issue.k8s.io/492)), which would change its `replicas` field. We will not add scheduling policies (e.g., [spreading](http://issue.k8s.io/367#issuecomment-48428019)) to the replication controller. Nor should it verify that the pods controlled match the currently specified template, as that would obstruct auto-sizing and other automated processes. Similarly, completion deadlines, ordering dependencies, configuration expansion, and other features belong elsewhere. We even plan to factor out the mechanism for bulk pod creation ([#170](http://issue.k8s.io/170)).

The replication controller is intended to be a composable building-block primitive. We expect higher-level APIs and/or tools to be built on top of it and other complementary primitives for user convenience in the future. The "macro" operations currently supported by kubectl (run, stop, scale, rolling-update) are proof-of-concept examples of this. For instance, we could imagine something like [Asgard](http://techblog.netflix.com/2012/06/asgard-web-based-cloud-management-and.html) managing replication controllers, auto-scalers, services, scheduling policies, canaries, etc.


## API Object

Replication controller is a top-level resource in the kubernetes REST API. More details about the
API object can be found at: [ReplicationController API
object](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/release-1.2/docs/api-reference/v1/definitions.html#_v1_replicationcontroller).

## Alternatives to Replication Controller

### Bare Pods

Unlike in the case where a user directly created pods, a replication controller replaces pods that are deleted or terminated for any reason, such as in the case of node failure or disruptive node maintenance, such as a kernel upgrade. For this reason, we recommend that you use a replication controller even if your application requires only a single pod. Think of it similarly to a process supervisor, only it supervises multiple pods across multiple nodes instead of individual processes on a single node.  A replication controller delegates local container restarts to some agent on the node (e.g., Kubelet or Docker).

### Job

Use a [Job](jobs.md) instead of a replication controller for pods that are expected to terminate on their own
(i.e. batch jobs).

### DaemonSet

Use a [DaemonSet](../admin/daemons.md) instead of a replication controller for pods that provide a
machine-level function, such as machine monitoring or machine logging.  These pods have a lifetime is tied
to machine lifetime: the pod needs to be running on the machine before other pods start, and are
safe to terminate when the machine is otherwise ready to be rebooted/shutdown.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/replication-controller.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
