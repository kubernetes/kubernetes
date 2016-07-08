# Background Rescheduling in Kubernetes

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Background Rescheduling in Kubernetes](#background-rescheduling-in-kubernetes)
  - [Overview](#overview)
  - [Objectives](#objectives)
  - [Algorithm](#algorithm)
  - [Rescheduling](#rescheduling)
  - [Disruption Budgets](#disruption-budgets)
  - [Implementation](#implementation)

<!-- END MUNGE: GENERATED_TOC -->

## Overview

There has been much discussion over the potential design and implementation of a rescheduler ([#12140](https://github.com/kubernetes/kubernetes/issues/12140), [#22217](https://github.com/kubernetes/kubernetes/issues/22217)). Most of this discussion has focused on allowing the scheduler to preempt pods during its scheduling operations. However, a second goal of a complete rescheduler is to optimize cluster layout by rescheduling nodes independently of the scheduler. Despite the apparent similarities, these two functions differ so significantly, both in design and probable implementation, that they can be designed and developed separately..

## Objectives

Generally speaking, the objective of the rescheduler is to make a best-effort attempt to further optimize the scheduling of pods. More specifically, we seek to improve nonoptimal conditions which arise as a result of changing conditions since the time of scheduling. We are primarily concerned with two types of of changes: the death of pods and the addition of new nodes to the cluster. This task, much like the task of scheduling itself, cannot be fully evaluated by a single simple metric. While the most dominant objective is to increase overall utilization, other conditions taken into account during scheduling, such as affinities/anti-affinities, must also be considered.

## Algorithm

Previous discussions of a background rescheduler have described a *global fitness function*: a function, probably implicit, which maps potential cluster layouts to numeric 'goodness' values. A simpler and more elegant solution arises from considering the actual problem more closely. The need for a rescheduler stems from the fact that the scheduler's decisions frequently become non-optimal as pods complete or nodes are added. Therefore, we want to identify those pods for which their current position differs from the , evict them, and allow the scheduler to schedule them again.

Our rescheduler, running in the background, will make a continual pass of all pods on the cluster in a random order. For each pod, it will run a modified version of the original scheduling algorithm. This algorithm will calculate priorities for all nodes in the cluster as if the pod was not already present on any node. If the priority of another node is greater than the priority of the pod's current node plus some stability threshold (to account for the cost of the rescheduling), the rescheduler will begin the process of moving the pod.

Some attention must be given to the modified scheduling algorithm. As the modified algorithm is technically decoupled from the scheduler's algorithm, there might be a temptation to use different sets of priorities and predicates. However, such a diversion would carry several severe disadvantages. Any non-trivial change to the algorithm would create some number of disagreements between the scheduler and the rescheduler. Such pods would be immediately reassigned and rescheduled as soon as the rescheduler examines the particular pod, effectively bypassing the scheduling algorithm entirely. There are several cases where that might be more desirable: such as a more advanced algorithm which cannot be implemented on the latency-sensitive scheduler. However, such an approach would likely incur the significantly expensive task of rescheduling an inordinate number of times.

## Rescheduling

The most basic procedure to accomplish the actual rescheduling would be to simply kill a pod, leaving the scheduling of its replacement to the replication controller and the scheduler. This approach requires minimal modification to other components, maintains the architecture of Kubernetes and avoids duplicating code.

There are two potential modifications to the procedure. The first is to handle the scheduling in the rescheduler itself. In order to prevent concurrency issues, the rescheduler would then submit the pod to the scheduler while indicating the precise node on which the pod would be scheduled. If the node in question is no longer available, the scheduler would find a new node using its own algorithm. This approach minimizes duplicated code, especially if the scheduling algorithm is already duplicated in the rescheduler, and would very slightly decrease scheduler latency. This approach is primarily useful in cases where the rescheduler's algorithm differs from that of the scheduler, and we do not expect it to be significantly useful.

The second potential modification would take advantage of the lack of time pressure in a background rescheduler. In this proposal, the rescheduler would first create and schedule (either by itself or through the standard scheduler) an additional duplicate pod. Only when the duplicate pod has been scheduled and passes readiness checks (or after a timeout period has passed) would the rescheduler delete the original pod, using the appropriate graceful procedures. Of course, the replication controller must be prevented from killing the duplicate pod.

This approach would minimize the disruption of rescheduling. In particular, previous designs would require each service to request an additional pod to account for disruptions during rescheduling. This approach would avoid that requirement, as services will never have less than their requested number of pods active (barring unplanned failures), increasing effective cluster utilization.

Either or both of these approaches might prove beneficial, but further investigation is needed to refine the precise movement process.

## Disruption Budgets

While disruption budgets and quotas, as described in #22217, are necessary for the effective design of a preemptive scheduler, they are not as essential to the design of a background rescheduler. A preemptive scheduler inevitably faces the task of prioritizing two different pods: it will harm the pod which is evicted and rescheduled while benefiting the pod which takes its place. This effect is especially pronounced if a pod must be moved onto a less-optimal node or  cannot be rescheduled altogether. The rescheduler, by contrast, *should *benefit, not harm, all pods with which it interacts.

Of course, this statement carries several caveats. In many cases, rescheduling trades a long term benefit (A more optimal node) for a short term loss (The costs of rescheduling). In other cases, the higher priority of a pod for a new node might be due to priority functions which optimize for the future placement of pods. Then, the pod would see little benefit and a significant cost to move. These effects would be drastically mitigated by the implementation of more graceful rescheduling, as described above.

It is also important to acknowledge that some services may not want to be rescheduled at all, or may be more hesitant towards rescheduling. These services include stateful pods which are not willing or able to checkpoint or transfer their state. To accommodate these services, we introduce two new annotations on pods. The first annotation will prevent any rescheduling of this pod, while the second will allow a custom threshold needed to justify rescheduling, allowing services to be rescheduled less (or more) frequently. However, as rescheduling is expected to be beneficial for the majority of pods in the majority of cases, no quotas or limits on these annotations should be required.

## Implementation

One unanswered question is the precise nature of the implementation. A rescheduler could potentially be implemented either as its own component, with a similar structure to the horizontal pod autoscaler, or as a separate thread in the scheduler. The rescheduler needs access to both the configuration and code (especially the priorities and predicates) of the scheduler and the access to a more complete API (beyond the `/bindings` access given to the scheduler). Neither implementations are entirely satisfactory.

