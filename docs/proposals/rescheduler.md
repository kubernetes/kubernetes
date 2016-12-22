# Rescheduler design space

@davidopp, @erictune, @briangrant

July 2015

## Introduction and definition

A rescheduler is an agent that proactively causes currently-running
Pods to be moved, so as to optimize some objective function for
goodness of the layout of Pods in the cluster. (The objective function
doesn't have to be expressed mathematically; it may just be a
collection of ad-hoc rules, but in principle there is an objective
function. Implicitly an objective function is described by the
scheduler's predicate and priority functions.) It might be triggered
to run every N minutes, or whenever some event happens that is known
to make the objective function worse (for example, whenever any Pod goes
PENDING for a long time.)

## Motivation and use cases

A rescheduler is useful because without a rescheduler, scheduling
decisions are only made at the time Pods are created. But later on,
the state of the cell may have changed in some way such that it would
be better to move the Pod to another node.

There are two categories of movements a rescheduler might trigger: coalescing
and spreading.

### Coalesce Pods

This is the most common use case. Cluster layout changes over time. For
example, run-to-completion Pods terminate, producing free space in their wake, but that space
is fragmented. This fragmentation might prevent a PENDING Pod from scheduling
(there are enough free resource for the Pod in aggregate across the cluster,
but not on any single node). A rescheduler can coalesce free space like a
disk defragmenter, thereby producing enough free space on a node for a PENDING
Pod to schedule. In some cases it can do this just by moving Pods into existing
holes, but often it will need to evict (and reschedule) running Pods in order to
create a large enough hole.

A second use case for a rescheduler to coalesce pods is when it becomes possible
to support the running Pods on a fewer number of nodes. The rescheduler can
gradually move Pods off of some set of nodes to make those nodes empty so
that they can then be shut down/removed. More specifically,
the system could do a simulation to see whether after removing a node from the
cluster, will the Pods that were on that node be able to reschedule,
either directly or with the help of the rescheduler; if the answer is
yes, then you can safely auto-scale down (assuming services will still
meeting their application-level SLOs).

### Spread Pods

The main use cases for spreading Pods revolve around relieving congestion on (a) highly
utilized node(s). For example, some process might suddenly start receiving a significantly
above-normal amount of external requests, leading to starvation of best-effort
Pods on the node. We can use the rescheduler to move the best-effort Pods off of the
node. (They are likely to have generous eviction SLOs, so are more likely to be movable
than the Pod that is experiencing the higher load, but in principle we might move either.)
Or even before any node becomes overloaded, we might proactively re-spread Pods from nodes
with high-utilization, to give them some buffer against future utilization spikes. In either
case, the nodes we move the Pods onto might have been in the system for a long time or might
have been added by the cluster auto-scaler specifically to allow the rescheduler to
rebalance utilization.

A second spreading use case is to separate antagonists.
Sometimes the processes running in two different Pods on the same node
may have unexpected antagonistic
behavior towards one another. A system component might monitor for such
antagonism and ask the rescheduler to move one of the antagonists to a new node.

### Ranking the use cases

The vast majority of users probably only care about rescheduling for three scenarios:

1. Move Pods around to get a PENDING Pod to schedule
1. Redistribute Pods onto new nodes added by a cluster auto-scaler when there are no PENDING Pods
1. Move Pods around when CPU starvation is detected on a node

## Design considerations and design space

Because rescheduling is disruptive--it causes one or more
already-running Pods to die when they otherwise wouldn't--a key
constraint on rescheduling is that it must be done subject to
disruption SLOs. There are a number of ways to specify these SLOs--a
global rate limit across all Pods, a rate limit across a set of Pods
defined by some particular label selector, a maximum number of Pods
that can be down at any one time among a set defined by some
particular label selector, etc. These policies are presumably part of
the Rescheduler's configuration.

There are a lot of design possibilities for a rescheduler. To explain
them, it's easiest to start with the description of a baseline
rescheduler, and then describe possible modifications. The Baseline
rescheduler
* only kicks in when there are one or more PENDING Pods for some period of time; its objective function is binary: completely happy if there are no PENDING Pods, and completely unhappy if there are PENDING Pods; it does not try to optimize for any other aspect of cluster layout
* is not a scheduler -- it simply identifies a node where a PENDING Pod could fit if one or more Pods on that node were moved out of the way, and then kills those Pods to make room for the PENDING Pod, which will then be scheduled there by the regular scheduler(s).  [obviously this killing operation must be able to specify "don't allow the killed Pod to reschedule back to whence it was killed" otherwise the killing is pointless] Of course it should only do this if it is sure the killed Pods will be able to reschedule into already-free space in the cluster. Note that although it is not a scheduler, the Rescheduler needs to be linked with the predicate functions of the scheduling algorithm(s) so that it can know (1) that the PENDING Pod would actually schedule into the hole it has identified once the hole is created, and (2) that the evicted Pod(s) will be able to schedule somewhere else in the cluster.

Possible variations on this Baseline rescheduler are

1. it can kill the Pod(s) whose space it wants **and also schedule the Pod that will take that space and reschedule the Pod(s) that were killed**, rather than just killing the Pod(s) whose space it wants and relying on the regular scheduler(s) to schedule the Pod that will take that space (and to reschedule the Pod(s) that were evicted)
1. it can run continuously in the background to optimize general cluster layout instead of just trying to get a PENDING Pod to schedule
1. it can try to move groups of Pods instead of using a one-at-a-time / greedy approach
1. it can formulate multi-hop plans instead of single-hop

A key design question for a Rescheduler is how much knowledge it needs about the scheduling policies used by the cluster's scheduler(s).
* For the Baseline rescheduler, it needs to know the predicate functions used by the cluster's scheduler(s) else it can't know how to create a hole that the PENDING Pod will fit into, nor be sure that the evicted Pod(s) will be able to reschedule elsewhere.
* If it is going to run continuously in the background to optimize cluster layout but is still only going to kill Pods, then it still needs to know the predicate functions for the reason mentioned above. In principle it doesn't need to know the priority functions; it could just randomly kill Pods and rely on the regular scheduler to put them back in better places. However, this is a rather inexact approach. Thus it is useful for the rescheduler to know the priority functions, or at least some subset of them, so it can be sure that an action it takes will actually improve the cluster layout.
* If it is going to run continuously in the background to optimize cluster layout and is going to act as a scheduler rather than just killing Pods, then it needs to know the predicate functions and some compatible (but not necessarily identical) priority functions  One example of a case where "compatible but not identical" might be useful is if the main scheduler(s) has a very simple scheduling policy optimized for low scheduling latency, and the Rescheduler having a more sophisticated/optimal scheduling policy that requires more computation time. The main thing to avoid is for the scheduler(s) and rescheduler to have incompatible priority functions, as this will cause them to "fight" (though it still can't lead to an infinite loop, since the scheduler(s) only ever touches a Pod once).

## Appendix: Integrating rescheduler with cluster auto-scaler (scale up)

For scaling up the cluster, a reasonable workflow might be:

1. pod horizontal auto-scaler decides to add one or more Pods to a service, based on the metrics it is observing
1. the Pod goes PENDING due to lack of a suitable node with sufficient resources
1. rescheduler notices the PENDING Pod and determines that the Pod cannot schedule just by rearranging existing Pods (while respecting SLOs)
1. rescheduler triggers cluster auto-scaler to add a node of the appropriate type for the PENDING Pod
1. the PENDING Pod schedules onto the new node (and possibly the rescheduler also moves other Pods onto that node)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/rescheduler.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
