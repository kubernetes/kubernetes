# Controlled Rescheduling in Kubernetes

## Overview

Although the Kubernetes scheduler(s) try to make good placement decisions for pods,
conditions in the cluster change over time (e.g. jobs finish and new pods arrive, nodes
are removed due to failures or planned maintenance or auto-scaling down, nodes appear due
to recovery after a failure or re-joining after maintenance or auto-scaling up or adding
new hardware to a bare-metal cluster), and schedulers are not omniscient (e.g. there are
some interactions between pods, or between pods and nodes, that they cannot predict). As
a result, the initial node selected for a pod may turn out to be a bad match, from the
perspective of the pod and/or the cluster as a whole, at some point after the pod has
started running.

Today (Kubernetes version 1.2) once a pod is scheduled to a node, it never moves unless
it terminates on its own, is deleted by the user, or experiences some unplanned event
(e.g. the node where it is running dies). Thus in a cluster with long-running pods, the
assignment of pods to nodes degrades over time, no matter how good an initial scheduling
decision the scheduler makes. This observation motivates "controlled rescheduling," a
mechanism by which Kubernetes will "move" already-running pods over time to improve their
placement. Controlled rescheduling is the subject of this proposal.

Note that the term "move" is not technically accurate -- the mechanism used is that
Kubernetes will terminate a pod that is managed by a controller, and the controller will
create a replacement pod that is then scheduled by the pod's scheduler. The terminated
pod and replacement pod are completely separate pods, and no pod migration is
implied. However, describing the process as "moving" the pod is approximately accurate
and easier to understand, so we will use this terminology in the document.

We use the term "rescheduling" to describe any action the system takes to move an
already-running pod. The decision may be made and executed by any component; we wil
introduce the concept of a "rescheduler" component later, but it is not the only
component that can do rescheduling.

This proposal primarily focuses on the architecture and features/mechanisms used to
achieve rescheduling, and only briefly discuss example policies. We expect that community
experimentation will lead to a significantly better understanding of the range, potential,
and limitations of rescheduling policies.

## Example use cases

Example use cases for rescheduling are

* moving a running pod onto a node that better satisfies its scheduling criteria
  * moving a pod onto an under-utilized node
  * moving a pod onto a node that meets more of the pod's affinity/anti-affinity preferences
* moving a running pod off of a node in anticipation of a known or speculated future event
  * draining a node in preparation for maintenance, decomissioning, auto-scale-down, etc.
  * "preempting" a running pod to make room for a pending pod to schedule
  * proactively/speculatively make room for large and/or exclusive pods to facilitate
    fast scheduling in the future (often called "defragmentation")
  * (note that these last two cases are the only use cases where the first-order intent
    is to move a pod specifically for the benefit of another pod)
* moving a running pod off of a node from which it is receiving poor service
  * anomalous crashlooping or other mysterious incompatiblity between the pod and the node
  * repeated out-of-resource killing (see #18724)
  * repeated attempts by the scheduler to schedule the pod onto some node, but it is
    rejected by Kubelet admission control due to incomplete scheduler knowledge
  * poor performance due to interference from other containers on the node (CPU hogs,
    cache thrashers, etc.) (note that in this case there is a choice of moving the victim
    or the aggressor)

## Some axes of the design space

Among the key design decisions are

* how does a pod specify its tolerance for these system-generated disruptions, and how
  does the system enforce such disruption limits
* for each use case, where is the decision made about when and which pods to reschedule
  (controllers, schedulers, an entirely new component e.g. "rescheduler", etc.)
* rescheduler design issues: how much does a rescheduler need to know about pods'
  schedulers' policies, how does the rescheduler specify its rescheduling
  requests/decisions (e.g. just as an eviction, an eviction with a hint about where to
  reschedule, or as an eviction paired with a specific binding), how does the system
  implement these requests, does the rescheduler take into account the second-order
  effects of decisions (e.g. whether an evicted pod will reschedule, will cause
  a preemption when it reschedules, etc.), does the rescheduler execute multi-step plans
  (e.g. evict two pods at the same time with the intent of moving one into the space
  vacated by the other, or even more complex plans)

Additional musings on the rescheduling design space can be found [here](rescheduler.md).

## Design proposal

The key mechanisms and components of the proposed design are priority, preemption,
disruption budgets, the `/evict` subresource, and the rescheduler.

### Priority

#### Motivation


Just as it is useful to overcommit nodes to increase node-level utilization, it is useful
to overcommit clusters to increase cluster-level utilization.  Scheduling priority (which
we abbreviate as *priority*, in combination with disruption budgets (described in the
next section), allows Kubernetes to safely overcommit clusters much as QoS levels allow
it to safely overcommit nodes.

Today, cluster sharing among users, workload types, etc. is regulated via the
[quota](../admin/resourcequota/README.md) mechanism. When allocating quota, a cluster
administrator has two choices: (1) the sum of the quotas is less than or equal to the
capacity of the cluster, or (2) the sum of the quotas is greater than the capacity of the
cluster (that is, the cluster is overcommitted).  (1) is likely to lead to cluster
under-utilization, while (2) is unsafe in the sense that someone's pods may go pending
indefinitely even though they are still within their quota. Priority makes cluster
overcommitment (i.e. case (2)) safe by allowing users and/or administrators to identify
which pods should be allowed to run, and which should go pending, when demand for cluster
resources exceeds supply to due to cluster overcommitment.

Priority is also useful in some special-case scenarios, such as ensuring that system
DaemonSets can always schedule and reschedule onto every node where they want to run
(assuming they are given the highest priority), e.g. see #21767.

#### Specifying priorities

We propose to add a required `Priority` field to `PodSpec`. Its value type is string, and
the cluster administrator defines a total ordering on these strings (for example
`Critical`, `Normal`, `Preemptible`). We choose string instead of integer so that it is
easy for an administrator to add new priority levels in between existing levels, to
encourage thinking about priority in terms of user intent and avoid magic numbers, and to
make the internal implementation more flexible.

When a scheduler is scheduling a new pod P and cannot find any node that meets all of P's
scheduling predicates, it is allowed to evict ("preempt") one or more pods that are at
the same or lower priority than P (subject to disruption budgets, see next section) from
a node in order to make room for P, i.e. in order to make the scheduling predicates
satisfied for P on that node.  (Note that when we add cluster-level resources (#19080),
it might be necessary to preempt from multiple nodes, but that scenario is outside the
scope of this document.)  The preempted pod(s) may or may not be able to reschedule. The
net effect of this process is that when demand for cluster resources exceeds supply, the
higher-priority pods will be able to run while the lower-priority pods will be forced to
wait. The detailed mechanics of preemption are described in a later section.

In addition to taking disruption budget into account, for equal-priority preemptions the
scheduler will try to enforce fairness (across victim controllers, services, etc.)

Priorities could be specified directly by users in the podTemplate, or assigned by an
admission controller using
properties of the pod. Either way, all schedulers must be configured to understand the
same priorities (names and ordering). This could be done by making them constants in the
API, or using ConfigMap to configure the schedulers with the information. The advantage of
the former (at least making the names, if not the ordering, constants in the API) is that
it allows the API server to do validation (e.g. to catch mis-spelling).

In the future, which priorities are usable for a given namespace and pods with certain
attributes may be configurable, similar to ResourceQuota, LimitRange, or security policy.

Priority and resource QoS are indepedent.

The priority we have described here might be used to prioritize the scheduling queue
(i.e. the order in which a scheduler examines pods in its scheduling loop), but the two
priority concepts do not have to be connected. It is somewhat logical to tie them
together, since a higher priority genreally indicates that a pod is more urgent to get
running. Also, scheduling low-priority pods before high-priority pods might lead to
avoidable preemptions if the high-priority pods end up preempting the low-priority pods
that were just scheduled.

TODO: Priority and preemption are global or namespace-relative? See
[this discussion thread](https://github.com/kubernetes/kubernetes/pull/22217#discussion_r55737389).

#### Relationship of priority to quota

Of course, if the decision of what priority to give a pod is solely up to the user, then
users have no incentive to ever request any priority less than the maximum.  Thus
priority is intimately related to quota, in the sense that resource quotas must be
allocated on a per-priority-level basis (X amount of RAM at priority A, Y amount of RAM
at priority B, etc.). The "guarantee" that highest-priority pods will always be able to
schedule can only be achieved if the sum of the quotas at the top priority level is less
than or equal to the cluster capacity. This is analogous to QoS, where safety can only be
achieved if the sum of the limits of the top QoS level ("Guaranteed") is less than or
equal to the node capacity. In terms of incentives, an organization could "charge"
an amount proportional to the priority of the resources.

The topic of how to allocate quota at different priority levels to achieve a desired
balance between utilization and probability of schedulability is an extremely complex
topic that is outside the scope of this document. For example, resource fragmentation and
RequiredDuringScheduling node and pod affinity and anti-affinity means that even if the
sum of the quotas at the top priority level is less than or equal to the total aggregate
capacity of the cluster, some pods at the top priority level might still go pending. In
general, priority provdes a *probabilistic* guarantees of pod schedulability in the face
of overcommitment, by allowing prioritization of which pods should be allowed to run pods
when demand for cluster resources exceeds supply.

### Disruption budget

While priority can protect pods from one source of disruption (preemption by a
lower-priority pod), *disruption budgets* limit disruptions from all Kubernetes-initiated
causes, including preemption by an equal or higher-priority pod, or being evicted to
achieve other rescheduling goals. In particular, each pod is optionally associated with a
"disruption budget," a new API resource that limits Kubernetes-initiated terminations
across a set of pods (e.g. the pods of a particular Service might all point to the same
disruption budget object), regardless of cause. Initially we expect disruption budget
(e.g. `DisruptionBudgetSpec`) to consist of

* a rate limit on disruptions (preemption and other evictions) across the corresponding
  set of pods, e.g. no more than one disruption per hour across the pods of a particular Service
* a minimum number of pods that must be up simultaneously (sometimes called "shard
  strength") (of course this can also be expressed as the inverse, i.e. the number of
  pods of the collection that can be down simultaneously)

The second item merits a bit more explanation. One use case is to specify a quorum size,
e.g. to ensure that at least 3 replicas of a quorum-based service with 5 replicas are up
at the same time. In practice, a service should ideally create enough replicas to survive
at least one planned and one unplanned outage. So in our quorum example, we would specify
that at least 4 replicas must be up at the same time; this allows for one intentional
disruption (bringing the number of live replicas down from 5 to 4 and consuming one unit
of shard strength budget) and one unplanned disruption (bringing the number of live
replicas down from 4 to 3) while still maintaining a quorum. Shard strength is also
useful for simpler replicated services; for example, you might not want more than 10% of
your front-ends to be down at the same time, so as to avoid overloading the remaining
replicas.

Initially, disruption budgets will be specified by the user. Thus as with priority,
disruption budgets need to be tied into quota, to prevent users from saying none of their
pods can ever be disrupted. The exact way of expressing and enforcing this quota is TBD,
though a simple starting point would be to have an admission controller assign a default
disruption budget based on priority level (more liberal with decreasing priority).
We also likely need a quota that applies to Kubernetes *components*, to the limit the rate
at which any one component is allowed to consume disruption budget.

Of course there should also be a `DisruptionBudgetStatus` that indicates the current
disruption rate that the collection of pods is experiencing, and the number of pods that
are up.

For the purposes of disruption budget, a pod is considered to be disrupted as soon as its
graceful termination period starts.

A pod that is not covered by a disruption budget but is managed by a controller,
gets an implicit disruption budget of infinite (though the system should try to not
unduly victimize such pods). How a pod that is not managed by a controller is
handled is TBD.

TBD: In addition to `PodSpec`, where do we store pointer to disruption budget
(podTemplate in controller that managed the pod?)? Do we auto-generate a disruption
budget (e.g. when instantiating a Service), or require the user to create it manually
before they create a controller? Which objects should return the disruption budget object
as part of the output on `kubectl get` other than (obviously) `kubectl get` for the
disruption budget itself?

TODO: Clean up distinction between "down due to voluntary action taken by Kubernetes"
and "down due to unplanned outage" in spec and status.

For now, there is nothing to prevent clients from circumventing the disruption budget
protections. Of course, clients that do this are not being "good citizens." In the next
section we describe a mechanism that at least makes it easy for well-behaved clients to
obey the disruption budgets.

See #12611 for additional discussion of disruption budgets.

### /evict subresource and PreferAvoidPods

Although we could put the responsibility for checking and updating disruption budgets
solely on the client, it is safer and more convenient if we implement that functionality
in the API server. Thus we will introduce a new `/evict` subresource on pod. It is similar to
today's "delete" on pod except

  * It will be rejected if the deletion would violate disruption budget. (See how
    Deployment handles failure of /rollback for ideas on how clients could handle failure
    of `/evict`.) There are two possible ways to implement this:

    * For the initial implementation, this will be accomplished by the API server just
    looking at the `DisruptionBudgetStatus` and seeing if the disruption would violate the
    `DisruptionBudgetSpec`. In this approach, we assume a disruption budget controller
    keeps the `DisruptionBudgetStatus` up-to-date by observing all pod deletions and
    creations in the cluster, so that an approved disruption is quickly reflected in the
    `DisruptionBudgetStatus`. Of course this approach does allow a race in which one or
    more additional disruptions could be approved before the first one is reflected in the
    `DisruptionBudgetStatus`.

    * Thus a subsequent implementation will have the API server explicitly debit the
    `DisruptionBudgetStatus` when it accepts an `/evict`. (There still needs to be a
    controller, to keep the shard strength status up-to-date when replacement pods are
    created after an eviction; the controller may also be necessary for the rate status
    depending on how rate is represented, e.g. adding tokens to a bucket at a fixed rate.)
    Once etcd support multi-object transactions (etcd v3), the debit and pod deletion will
    be placed in the same transaction.

    * Note: For the purposes of disruption budget, a pod is considered to be disrupted as soon as its
    graceful termination period starts (so when we say "delete" here we do not mean
    "deleted from etcd" but rather "graceful termination period has started").

  * It will allow clients to communicate additional parameters when they wish to delete a
  pod. (In the absence of the `/evict` subresource, we would have to create a pod-specific
  type analogous to `api.DeleteOptions`.)

We will make `kubectl delete pod` use `/evict` by default, and require a command-line
flag to delete the pod directly.

We will add to `NodeStatus` a bounded-sized list of signatures of pods that should avoid
that node (provisionally called `PreferAvoidPods`). One of the pieces of information
specified in the `/evict` subresource is whether the eviction should add the evicted
pod's signature to the corresponding node's `PreferAvoidPods`. Initially the pod
signature will be a
[controllerRef](https://github.com/kubernetes/kubernetes/issues/14961#issuecomment-183431648),
i.e. a reference to the pod's controller. Controllers are responsible for garbage
collecting, after some period of time, `PreferAvoidPods` entries that point to them, but the API
server will also enforce a bounded size on the list. All schedulers will have a
highest-weighted priority function that gives a node the worst priority if the pod it is
scheduling appears in that node's `PreferAvoidPods` list. Thus appearing in
`PreferAvoidPods` is similar to
[RequiredDuringScheduling node anti-affinity](../../docs/user-guide/node-selection/README.md)
but it takes precedence over all other priority criteria and is not explicitly listed in
the `NodeAffinity` of the pod.

`PreferAvoidPods` is useful for the "moving a running pod off of a node from which it is
receiving poor service" use case, as it reduces the chance that the replacement pod will
end up on the same node (keep in mind that most of those cases are situations that the
scheduler does not have explicit priority functions for, for example it cannot know in
advance that a pod will be starved). Also, though we do not intend to implement any such
policies in the first version of the rescheduler, it is useful whenever the rescheduler evicts
two pods A and B with the intention of moving A into the space vacated by B (it prevents
B from rescheduling back into the space it vacated before A's scheduler has a chance to
reschedule A there). Note that these two uses are subtly different; in the first
case we want the avoidance to last a relatively long time, whereas in the second case we
may only need it to last until A schedules.

See #20699 for more discussion.

### Preemption mechanics

**NOTE: We expect a fuller design doc to be written on preemption before it is implemented.
However, a sketch of some ideas are presented here, since preemption is closely related to the
concepts discussed in this doc.**

Pod schedulers will decide and enact preemptions, subject to the priority and disruption
budget rules described earlier. (Though note that we currently do not have any mechanism
to prevent schedulers from bypassing either the priority or disruption budget rules.)
The scheduler does not concern itself with whether the evicted pod(s) can reschedule. The
eviction(s) use(s) the `/evict` subresource so that it is subject to the disruption
budget(s) of the victim(s), but it does not request to add the victim pod(s) to the
nodes' `PreferAvoidPods`.

Evicting victim(s) and binding the pending pod that the evictions are intended to enable
to schedule, are not transactional. We expect the scheduler to issue the operations in
sequence, but it is still possible that another scheduler could schedule its pod in
between the eviction(s) and the binding, or that the set of pods running on the node in
question changed between the time the scheduler made its decision and the time it sent
the operations to the API server thereby causing the eviction(s) to be not sufficient to get the
pending pod to schedule. In general there are a number of race conditions that cannot be
avoided without (1) making the evictions and binding be part of a single transaction, and
(2) making the binding preconditioned on a version number that is associated with the
node and is incremented on every binding. We may or may not implement those mechanisms in
the future.

Given a choice between a node where scheduling a pod requires preemption and one where it
does not, all other things being equal, a scheduler should choose the one where
preemption is not required. (TBD: Also, if the selected node does require preemption, the
scheduler should preempt lower-priority pods before higher-priority pods (e.g. if the
scheduler needs to free up 4 GB of RAM, and the node has two 2 GB low-priority pods and
one 4 GB high-priority pod, all of which have sufficient disruption budget, it should
preempt the two low-priority pods). This is debatable, since all have sufficient
disruption budget. But still better to err on the side of giving better disruption SLO to
higher-priority pods when possible?)

Preemption victims must be given their termination grace period. One possible sequence
of events is

1. The API server binds the preemptor to the node (i.e. sets `nodeName` on the
preempting pod) and sets `deletionTimestamp` on the victims
2. Kubelet sees that `deletionTimestamp` has been set on the victims; they enter their
graceful termination period
3. Kubelet sees the preempting pod. It runs the admission checks on the new pod
assuming all pods that are in their graceful termination period are gone and that
all pods that are in the waiting state (see (4)) are running.
4. If (3) fails, then the new pod is rejected. If (3) passes, then Kubelet holds the
new pod in a waiting state, and does not run it until the pod passes passes the
admission checks using the set of actually running pods.

Note that there are a lot of details to be figured out here; above is just a very
hand-wavy sketch of one general approach that might work.

See #22212 for additional discussion.

### Node drain

Node drain will be handled by one or more components not described in this document. They
will respect disruption budgets. Initially, we will just make `kubectl drain`
respect disruption budgets.  See #17393 for other discussion.

### Rescheduler

All rescheduling other than preemption and node drain will be decided and enacted by a
new component called the *rescheduler*. It runs continuously in the background, looking
for opportunities to move pods to better locations. It acts when the degree of
improvement meets some threshold and is allowed by the pod's disruption budget.  The
action is eviction of a pod using the `/evict` subresource, with the pod's signature
enqueued in the node's `PreferAvoidPods`. It does not force the pod to reschedule to any
particular node. Thus it is really an "unscheduler"; only in combination with the evicted
pod's scheduler, which schedules the replacement pod, do we get true "rescheduling."  See
the "Example use cases" section earlier for some example use cases.

The rescheduler is a best-effort service that makes no guarantees about how quickly (or
whether) it will resolve a suboptimal pod placement.

The first version of the rescheduler will not take into consideration where or whether an
evicted pod will reschedule. The evicted pod may go pending, consuming one unit of the
corresponding shard strength disruption budget by one indefinitely. By using the `/evict`
subresource, the rescheduler ensures that an evicted pod has sufficient budget for the
evicted pod to go and stay pending.  We expect future versions of the rescheduler may be
linked with the "mandatory" predicate functions (currently, the ones that constitute the
Kubelet admission criteria), and will only evict if the rescheduler determines that the
pod can reschedule somewhere according to those criteria. (Note that this still does not
guarantee that the pod actually will be able to reschedule, for at least two reasons: (1)
the state of the cluster may change between the time the rescheduler evaluates it and
when the evicted pod's scheduler tries to schedule the replacement pod, and (2) the
evicted pod's scheduler may have additional predicate functions in addition to the
mandatory ones).

(Note: see [this comment](https://github.com/kubernetes/kubernetes/pull/22217#discussion_r54527968)).

The first version of the rescheduler will only implement two objectives: moving a pod
onto an under-utilized node, and moving a pod onto a node that meets more of the pod's
affinity/anti-affinity preferences than wherever it is currently running. (We assume that
nodes that are intentionally under-utilized, e.g. because they are being drained, are
marked unschedulable, thus the first objective will not cause the rescheduler to "fight"
a system that is draining nodes.)  We assume that all schedulers sufficiently weight the
priority functions for affinity/anti-affinity and avoiding very packed nodes,
otherwise evicted pods may not actually move onto a node that is better according to
the criteria that caused it to be evicted. (But note that in all cases it will move to a
node that is better according to the totality of its scheduler's priority functions,
except in the case where the node where it was already running was the only node
where it can run.) As a general rule, the rescheduler should only act when it sees
particularly bad situations, since (1) an eviction for a marginal improvement is likely
not worth the disruption--just because there is sufficient budget for an eviction doesn't
mean an eviction is painless to the application, and (2) rescheduling the pod might not
actually mitigate the identified problem if it is minor enough that other scheduling
factors dominate the decision of where the replacement pod is scheduled.

We assume schedulers' priority functions are at least vaguely aligned with the
rescheduler's policies; otherwise the rescheduler will never accomplish anything useful,
given that it relies on the schedulers to actually reschedule the evicted pods. (Even if
the rescheduler acted as a scheduler, explicitly rebinding evicted pods, we'd still want
this to be true, to prevent the schedulers and rescheduler from "fighting" one another.)

The rescheduler will be configured using ConfigMap; the cluster administrator can enable
or disable policies and can tune the rescheduler's aggressiveness (aggressive means it
will use a relatively low threshold for triggering an eviction and may consume a lot of
disruption budget, while non-aggressive means it will use a relatively high threshold for
triggering an eviction and will try to leave plenty of buffer in disruption budgets). The
first version of the rescheduler will not be extensible or pluggable, since we want to
keep the code simple while we gain experience with the overall concept. In the future, we
anticipate a version that will be extensible and pluggable.

We might want some way to force the evicted pod to the front of the scheduler queue,
independently of its priority.

See #12140 for additional discussion.

### Final comments

In general, the design space for this topic is huge. This document describes some of the
design considerations and proposes one particular initial implementation. We expect
certain aspects of the design to be "permanent" (e.g. the notion and use of priorities,
preemption, disruption budgets, and the `/evict` subresource) while others may change over time
(e.g. the partitioning of functionality between schedulers, controllers, rescheduler,
horizontal pod autoscaler, and cluster autoscaler; the policies the rescheduler implements;
the factors the rescheduler takes into account when making decisions (e.g. knowledge of
schedulers' predicate and priority functions, second-order effects like whether and where
evicted pod will be able to reschedule, etc.); the way the rescheduler enacts its
decisions; and the complexity of the plans the rescheduler attempts to implement).

## Implementation plan

The highest-priority feature to implement is the rescheduler with the two use cases
highlighted earlier: moving a pod onto an under-utilized node, and moving a pod onto a
node that meets more of the pod's affinity/anti-affinity preferences.  The former is
useful to rebalance pods after cluster auto-scale-up, and the latter is useful for
Ubernetes. This requires implementing disruption budgets and the `/evict` subresource,
but not priority or preemption.

Because the general topic of rescheduling is very speculative, we have intentionally
proposed that the first version of the rescheduler be very simple -- only uses eviction
(no attempt to guide replacement pod to any particular node), doesn't know schedulers'
predicate or priority functions, doesn't try to move two pods at the same time, and only
implements two use cases. As alluded to in the previous subsection, we expect the design
and implementation to evolve over time, and we encourage members of the community to
experiment with more sophisticated policies and to report their results from using them
on real workloads.

## Alternative implementations

TODO.

## Additional references

TODO.

TODO: Add reference to this doc from docs/proposals/rescheduler.md


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/rescheduling.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
