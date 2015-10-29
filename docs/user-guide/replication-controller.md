<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Replication Controller

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Replication Controller](#replication-controller)
  - [What is a _replication controller_?](#what-is-a-replication-controller)
  - [How does a replication controller work?](#how-does-a-replication-controller-work)
    - [Pod template](#pod-template)
    - [Labels](#labels)
  - [Responsibilities of the replication controller](#responsibilities-of-the-replication-controller)
  - [Common usage patterns](#common-usage-patterns)
    - [Rescheduling](#rescheduling)
    - [Scaling](#scaling)
    - [Rolling updates](#rolling-updates)
    - [Multiple release tracks](#multiple-release-tracks)
  - [API Object](#api-object)

<!-- END MUNGE: GENERATED_TOC -->

## What is a _replication controller_?

A _replication controller_ ensures that a specified number of pod "replicas" are running at any one time.  If there are too many, it will kill some.  If there are too few, it will start more. Unlike in the case where a user directly created pods, a replication controller replaces pods that are deleted or terminated for any reason, such as in the case of node failure or disruptive node maintenance, such as a kernel upgrade. For this reason, we recommend that you use a replication controller even if your application requires only a single pod. Think of it similarly to a process supervisor, only it supervises multiple pods across multiple nodes instead of individual processes on a single node.  A replication controller delegates local container restarts to some agent on the node (e.g., Kubelet or Docker).

As discussed in [life of a pod](pod-states.md), `ReplicationController` is *only* appropriate for pods with `RestartPolicy = Always`. (Note: If `RestartPolicy` is not set, the default value is `Always`.)  `ReplicationController` should refuse to instantiate any pod that has a different restart policy. As discussed in [issue #503](http://issue.k8s.io/503#issuecomment-50169443), we expect other types of controllers to be added to Kubernetes to handle other types of workloads, such as build/test and batch workloads, in the future.

A replication controller will never terminate on its own, but it isn't expected to be as long-lived as services. Services may be composed of pods controlled by multiple replication controllers, and it is expected that many replication controllers may be created and destroyed over the lifetime of a service (for instance, to perform an update of pods that run the service). Both services themselves and their clients should remain oblivious to the replication controllers that maintain the pods of the services.

## How does a replication controller work?

### Pod template

A replication controller creates new pods from a template, which is currently inline in the `ReplicationController` object, but which we plan to extract into its own resource [#170](http://issue.k8s.io/170).

Rather than specifying the current desired state of all replicas, pod templates are like cookie cutters. Once a cookie has been cut, the cookie has no relationship to the cutter. There is no quantum entanglement. Subsequent changes to the template or even switching to a new template has no direct effect on the pods already created. Similarly, pods created by a replication controller may subsequently be updated directly. This is in deliberate contrast to pods, which do specify the current desired state of all containers belonging to the pod. This approach radically simplifies system semantics and increases the flexibility of the primitive, as demonstrated by the use cases explained below.

Pods created by a replication controller are intended to be fungible and semantically identical, though their configurations may become heterogeneous over time. This is an obvious fit for replicated stateless servers, but replication controllers can also be used to maintain availability of master-elected, sharded, and worker-pool applications. Such applications should use dynamic work assignment mechanisms, such as the [etcd lock module](https://coreos.com/docs/distributed-configuration/etcd-modules/) or [RabbitMQ work queues](https://www.rabbitmq.com/tutorials/tutorial-two-python.html), as opposed to static/one-time customization of the configuration of each pod, which is considered an anti-pattern. Any pod customization performed, such as vertical auto-sizing of resources (e.g., cpu or memory), should be performed by another online controller process, not unlike the replication controller itself.

### Labels

The population of pods that a replication controller is monitoring is defined with a [label selector](labels.md#label-selectors), which creates a loosely coupled relationship between the controller and the pods controlled, in contrast to pods, which are more tightly coupled to their definition. We deliberately chose not to represent the set of pods controlled using a fixed-length array of pod specifications, because our experience is that approach increases complexity of management operations, for both clients and the system.

The replication controller should verify that the pods created from the specified template have labels that match its label selector. Though it isn't verified yet, you should also ensure that only one replication controller controls any given pod, by ensuring that the label selectors of replication controllers do not target overlapping sets. If you do end up with multiple controllers that have overlapping selectors, you will have to manage the deletion yourself with --cascade=false until there are no controllers with an overlapping superset of selectors.

Note that replication controllers may themselves have labels and would generally carry the labels their corresponding pods have in common, but these labels do not affect the behavior of the replication controllers.

Pods may be removed from a replication controller's target set by changing their labels. This technique may be used to remove pods from service for debugging, data recovery, etc. Pods that are removed in this way will be replaced automatically (assuming that the number of replicas is not also changed).

Similarly, deleting a replication controller using the API does not affect the pods it created. Its `replicas` field must first be set to `0` in order to delete the pods controlled. (Note that the client tool, `kubectl`, provides a single operation, [delete](kubectl/kubectl_delete.md) to delete both the replication controller and the pods it controls. If you want to leave the pods running when deleting a replication controller, specify `--cascade=false`. However, there is no such operation in the API at the moment)

## Responsibilities of the replication controller

The replication controller simply ensures that the desired number of pods matches its label selector and are operational. Currently, only terminated pods are excluded from its count. In the future, [readiness](http://issue.k8s.io/620) and other information available from the system may be taken into account, we may add more controls over the replacement policy, and we plan to emit events that could be used by external clients to implement arbitrarily sophisticated replacement and/or scale-down policies.

The replication controller is forever constrained to this narrow responsibility. It itself will not perform readiness nor liveness probes. Rather than performing auto-scaling, it is intended to be controlled by an external auto-scaler (as discussed in [#492](http://issue.k8s.io/492)), which would change its `replicas` field. We will not add scheduling policies (e.g., [spreading](http://issue.k8s.io/367#issuecomment-48428019)) to the replication controller. Nor should it verify that the pods controlled match the currently specified template, as that would obstruct auto-sizing and other automated processes. Similarly, completion deadlines, ordering dependencies, configuration expansion, and other features belong elsewhere. We even plan to factor out the mechanism for bulk pod creation ([#170](http://issue.k8s.io/170)).

The replication controller is intended to be a composable building-block primitive. We expect higher-level APIs and/or tools to be built on top of it and other complementary primitives for user convenience in the future. The "macro" operations currently supported by kubectl (run, stop, scale, rolling-update) are proof-of-concept examples of this. For instance, we could imagine something like [Asgard](http://techblog.netflix.com/2012/06/asgard-web-based-cloud-management-and.html) managing replication controllers, auto-scalers, services, scheduling policies, canaries, etc.

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

## API Object

Replication controller is a top-level resource in the kubernetes REST API. More details about the
API object can be found at: [ReplicationController API
object](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/v1.1.0/docs/api-reference/v1/definitions.html#_v1_replicationcontroller).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/replication-controller.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
