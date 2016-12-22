# Rescheduler: guaranteed scheduling of critical addons

## Motivation

In addition to Kubernetes core components like api-server, scheduler, controller-manager running on a master machine
there is a bunch of addons which due to various reasons have to run on a regular cluster node, not the master.
Some of them are critical to have fully functional cluster: Heapster, DNS, UI. Users can break their cluster
by evicting a critical addon (either manually or as a side effect of an other operation like upgrade)
which possibly can become pending (for example when the cluster is highly utilized).
To avoid such situation we want to have a mechanism which guarantees that
critical addons are scheduled assuming the cluster is big enough.
This possibly may affect other pods (including production user’s applications).

## Design

Rescheduler will ensure that critical addons are always scheduled.
In the first version it will implement only this policy, but later we may want to introduce other policies.
It will be a standalone component running on master machine similarly to scheduler.
Those components will share common logic (initially rescheduler will in fact import some of scheduler packages).

### Guaranteed scheduling of critical addons

Rescheduler will observe critical addons
(with annotation `scheduler.alpha.kubernetes.io/critical-pod`).
If one of them is marked by scheduler as unschedulable (pod condition `PodScheduled` set to `false`, the reason set to `Unschedulable`)
the component will try to find a space for the addon by evicting some pods and then the scheduler will schedule the addon.

#### Scoring nodes

Initially we want to choose a random node with enough capacity
(chosen as described in [Evicting pods](rescheduling-for-critical-pods.md#evicting-pods)) to schedule given addons.
Later we may want to introduce some heuristic:
* minimize number of evicted pods with violation of disruption budget or shortened termination grace period
* minimize number of affected pods by choosing a node on which we have to evict less pods
* increase probability of scheduling of evicted pods by preferring a set of pods with the smallest total sum of requests
* avoid nodes which are ‘non-drainable’ (according to drain logic), for example on which there is a pod which doesn’t belong to any RC/RS/Deployment

#### Evicting pods

There are 2 mechanism which possibly can delay a pod eviction: Disruption Budget and Termination Grace Period.

While removing a pod we will try to avoid violating Disruption Budget, though we can’t guarantee it
since there is a chance that it would block this operation for longer period of time.
We will also try to respect Termination Grace Period, though without any guarantee.
In case we have to remove a pod with termination grace period longer than 10s it will be shortened to 10s.

The proposed order while choosing a node to schedule a critical addon and pods to remove:
1. a node where the critical addon pod can fit after evicting only pods satisfying both
(1) their disruption budget will not be violated by such eviction and (2) they have grace period <= 10 seconds
1. a node where the critical addon pod can fit after evicting only pods whose disruption budget will not be violated by such eviction
1. any node where the critical addon pod can fit after evicting some pods

### Interaction with Scheduler

To avoid situation when Scheduler will schedule another pod into the space prepared for the critical addon,
the chosen node has to be temporarily excluded from a list of nodes considered by Scheduler while making decisions.
For this purpose the node will get a temporary
[Taint](../../docs/design/taint-toleration-dedicated.md) “CriticalAddonsOnly”
and each critical addon has to have defined toleration for this taint.
After Rescheduler has no more work to do: all critical addons are scheduled or cluster is too small for them,
all taints will be removed.

### Interaction with Cluster Autoscaler

Rescheduler possibly can duplicate the responsibility of Cluster Autoscaler:
both components are taking action when there is unschedulable pod.
It may cause the situation when CA will add extra node for a pending critical addon
and Rescheduler will evict some running pods to make a space for the addon.
This situation would be rare and usually an extra node would be anyway needed for evicted pods.
In the worst case CA will add and then remove the node.
To not complicate architecture by introducing interaction between those 2 components we accept this overlap.

We want to ensure that CA won’t remove nodes with critical addons by adding appropriate logic there.

### Rescheduler control loop

The rescheduler control loop will be as follow:
* while there is an unschedulable critical addon do the following:
  * choose a node on which the addon should be scheduled (as described in Evicting pods)
  * add taint to the node to prevent scheduler from using it
  * delete pods which blocks the addon from being scheduled
  * wait until scheduler will schedule the critical addon
* if there is no more critical addons for which we can help, ensure there is no node with the taint


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/rescheduling-for-critical-pods.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
