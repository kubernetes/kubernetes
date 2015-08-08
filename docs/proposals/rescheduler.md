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
[here](http://releases.k8s.io/release-1.0/docs/proposals/rescheduler.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Rescheduler design space

@davidopp, @erictune, @briangrant

July 2015

A rescheduler is an agent that proactively causes currently-running
Pods to be moved, so as to optimize some objective function for
goodness of the layout of Pods in the cluster. (The objective function
doesn't have to be expressed mathematically; it may just be a
collection of ad-hoc rules, but in principle there is an objective
function. Implicitly an objective function is described by the
scheduler's predicate and priority functions.) It might be triggered
to run every N minutes, or whenever some event happens that is known
to make the objective function worse (for example, whenever a Pod goes
PENDING for a long time.)

A rescheduler is useful because without a rescheduler, scheduling
decisions are only made at the time Pods are created. But as the
cluster layout changes over time, free "holes" are often produced that
were not available when a Pod was initially scheduled. These holes are
produced by run-to-completion Pods terminating, empty nodes being
added by a node auto-scaler, etc. Moving already-running Pods into
these holes may lead to a better cluster layout. A rescheduler might
not just exploit existing holes, but also create holes by evicting
Pods (assuming it knows they can reschedule elsewhere), as in free
space defragmentation.

[Although alluded to above, it's worth emphasizing that rescheduling
is the only way to make use of new nodes added by a cluster
auto-scaler (unless Pods were already PENDING; but even then, it's
likely advantageous to put more than just the previously PENDING Pods
on the new nodes.)]

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

The vast majority of users probably only care about rescheduling for three scenarios:

1. Redistribute Pods onto new nodes added by a cluster auto-scaler
1. Move Pods around to get a PENDING Pod to schedule
1. Move Pods around when CPU starvation is detected on a node

**Addendum: How a rescheduler might trigger cluster auto-scaling (to
scale up).** Instead of moving Pods around to free up space, it might
just add a new node (and then move some Pods onto the new node). More
generally, it might be useful to integrate the rescheduler and cluster
auto-scaler. For scaling up the cluster a reasonable workflow might
be:
1. pod horizontal auto-scaler decides to add one or more Pods to a service, based on the metrics it is observing
1. the Pod goes PENDING due to lack of a suitable node with sufficient resources
1. rescheduler notices the PENDING Pod and determines that the Pod cannot schedule just by rearranging existing Pods (while respecting SLOs)
1. rescheduler triggers cluster auto-scaler to add a node of the appropriate type for the PENDING Pod
1. the PENDING Pod schedules onto the new node (and possibly the rescheduler also moves other Pods onto that node)

**Addendum: Role of simulation.** Things like knowing what will be the
effect of different rearrangements of Pods requires a form of
simulation of the scheduling algorithm (see also discussion in
previous entry about what the rescheduler needs to know about the
predicate and priority functions of the cluster's scheduler(s)). For
cluster auto-scaling down, you could do a
simulation to see whether after removing a node from the cluster, will
the Pods that were on that node be able to reschedule, either directly
or with the help of the rescheduler; if the answer is yes, then you
can safely auto-scale down (assuming services will still meeting their
application-level SLOs).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/rescheduler.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
