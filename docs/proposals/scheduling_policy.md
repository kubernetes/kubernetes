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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/multiple-schedulers.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Round-robin Scheduling and Deadline Scheduling Proposal
----
***Status***: Design & Implementation in progress
> contact @combk8s or @mqliang

###Motivation
----
The current default scheduler of k8s insert all incoming pods which need to be scheduled into a simple FIFO queue, and schedule 
them one by one. The behavior is easy to understand. 

However, it is common that multiple types of workload, are running in the same cluster and they need to be scheduled in different
ways and have different scheduling requirements, for example some pods may need to be schedule ASAP, regardless how many pods are 
in the queue. In addition, the current scheduler have starvation drawbacks: for example, if there are 100 pods in namespace A and 
1 pod in namespace B waitting for scheduling, scheduler will not start to schedule the one pod in namespace B until all the 100 pods
in namespace A have been scheduled! 

To solve those problem, a more flexible sceduling policy will be very helpful. Thus we propose to introduce two scheduling policy 
***Round-Robin Scheduling*** and ***Deadline shcheduling***.

In addition, multiple-scheduler feature has been put on the table and is under heavy development. Once we introduce more scheduling
policy, user could deployemnt several scheduler with different scheduling policy, to meet different shceduling requirements.


###FIFO Scheduling Policy
----
FIFO scheduler will insert all incoming pods which need to be scheduled into a simple FIFO queue, and schedule them one by 
one. The behavior of the current default scheduler is like this.

###Round-Robin Scheduling Policy
----
Scheduler will place all the pods which need to be scheduled into a number of per-namespace queues. When scheduler decide which pod 
to schedule, it will cyclic between different namespace queue, pop a pod from one namespace a time, thus Round-Robin scheduling could 
ensure scheduling fair between namespaces.

###Deadline Shcheduling Policy
----
The main goal of the Deadline scheduling policy is to guarantee a start scheduling time for a pod. It does so by imposing a deadline
on pods to prevent starvation. Scheduler should maintain a deadline queue, pods in the deadline queue are basically sorted by their
enqueuing time, but once a pod's deadline(the exparation time) arrived, the pod will pop up to the top.

Before scheduling the next pod, the scheduler should decide which pod to schedule. If there is no pods in the queue expired, scheduler
will serve them by the order of enqueuing. Otherwise, pop the expired pod and schedule it immediately.

This feature will be very helpful for some high priority Pod wich need to be scheduled ASAP.


### API 
----

1) 

We will add

```
type SchedulePolicy string

const (
    SchedulerPolicyFIFO SchedulingPolicy = "FIFO"
    SchedulerPolicyRR   SchedulingPolicy = "RR"
)

```

to `KubeSchedulerConfiguration`, indicating the scheduler will apply FIFO or Round-Robin policy when it decide which pod to schedule
next.

2) 
We will add

```
ScheduleDeadline *time.Duration

```

to `PodSpec`, indicating the scheduling deadline 

### Implementation plan
----
1) Add the API describe above

2) Make scheduler respect SchedulePolicy

3) Implement Deadline scheduling functionality


### Future work
----
1) Extend the basic implementation of SchedulePolicy, such as add Weighted Round-Robin Scheduling policy and Priority Scheduling
Policy

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/multiple-schedulers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
