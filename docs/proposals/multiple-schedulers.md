# Multi-Scheduler in Kubernetes

**Status**: Design & Implementation in progress.

> Contact @HaiyangDING for questions & suggestions.

## Motivation

In current Kubernetes design, there is only one default scheduler in a Kubernetes cluster.
However it is common that multiple types of workload, such as traditional batch, DAG batch, streaming and user-facing production services,
are running in the same cluster and they need to be scheduled in different ways. For example, in
[Omega](http://research.google.com/pubs/pub41684.html) batch workload and service workload are scheduled by two types of schedulers:
the batch workload is scheduled by a scheduler which looks at the current usage of the cluster to improve the resource usage rate
and the service workload is scheduled by another one which considers the reserved resources in the
cluster and many other constraints since their performance must meet some higher SLOs.
[Mesos](http://mesos.apache.org/) has done a great work to support multiple schedulers by building a
two-level scheduling structure. This proposal describes how Kubernetes is going to support multi-scheduler
so that users could be able to run their user-provided scheduler(s) to enable some customized scheduling
behavior as they need. As previously discussed in [#11793](https://github.com/kubernetes/kubernetes/issues/11793),
[#9920](https://github.com/kubernetes/kubernetes/issues/9920) and [#11470](https://github.com/kubernetes/kubernetes/issues/11470),
the design of the multiple scheduler should be generic and includes adding a scheduler name annotation to separate the pods.
It is worth mentioning that the proposal does not address the question of how the scheduler name annotation gets
set although it is reasonable to anticipate that it would be set by a component like admission controller/initializer,
as the doc currently does.

Before going to the details of this proposal, below lists a number of the methods to extend the scheduler:

- Write your own scheduler and run it along with Kubernetes native scheduler. This is going to be detailed in this proposal
- Use the callout approach such as the one implemented in [#13580](https://github.com/kubernetes/kubernetes/issues/13580)
- Recompile the scheduler with a new policy
- Restart the scheduler with a new [scheduler policy config file](../../examples/scheduler-policy-config.json)
- Or maybe in future dynamically link a new policy into the running scheduler

## Challenges in multiple schedulers

- Separating the pods

    Each pod should be scheduled by only one scheduler. As for implementation, a pod should
    have an additional field to tell by which scheduler it wants to be scheduled. Besides,
    each scheduler, including the default one, should have a unique logic of how to add unscheduled
    pods to its to-be-scheduled pod queue. Details will be explained in later sections.

- Dealing with conflicts

    Different schedulers are essentially separated processes. When all schedulers try to schedule
    their pods onto the nodes, there might be conflicts.

    One example of the conflicts is resource racing: Suppose there be a `pod1` scheduled by
    `my-scheduler` requiring 1 CPU's *request*, and a `pod2` scheduled by `kube-scheduler` (k8s native
    scheduler, acting as default scheduler) requiring 2 CPU's *request*, while `node-a` only has 2.5
    free CPU's, if both schedulers all try to put their pods on `node-a`, then one of them would eventually
    fail when Kubelet on `node-a` performs the create action due to insufficient CPU resources.

    This conflict is complex to deal with in api-server and etcd. Our current solution is to let Kubelet
    to do the conflict check and if the conflict happens, effected pods would be put back to scheduler
    and waiting to be scheduled again. Implementation details are in later sections.

## Where to start: initial design

We definitely want the multi-scheduler design to be a generic mechanism. The following lists the changes
we want to make in the first step.

- Add an annotation in pod template: `scheduler.alpha.kubernetes.io/name: scheduler-name`, this is used to
separate pods between schedulers. `scheduler-name` should match one of the schedulers' `scheduler-name`
- Add a `scheduler-name` to each scheduler. It is done by hardcode or as command-line argument. The
Kubernetes native scheduler (now `kube-scheduler` process) would have the name as `kube-scheduler`
- The `scheduler-name` plays an important part in separating the pods between different schedulers.
Pods are statically dispatched to different schedulers based on `scheduler.alpha.kubernetes.io/name: scheduler-name`
annotation and there should not be any conflicts between different schedulers handling their pods, i.e. one pod must
NOT be claimed by more than one scheduler. To be specific, a scheduler can add a pod to its queue if and only if:
    1. The pod has no nodeName, **AND**
    2. The `scheduler-name` specified in the pod's annotation `scheduler.alpha.kubernetes.io/name: scheduler-name`
    matches the `scheduler-name` of the scheduler.

        The only one exception is the default scheduler. Any pod that has no `scheduler.alpha.kubernetes.io/name: scheduler-name`
        annotation is assumed to be handled by the "default scheduler". In the first version of the multi-scheduler feature,
        the default scheduler would be the Kubernetes built-in scheduler with `scheduler-name` as `kube-scheduler`.
        The Kubernetes build-in scheduler will claim any pod which has no `scheduler.alpha.kubernetes.io/name: scheduler-name`
        annotation or which has `scheduler.alpha.kubernetes.io/name: kube-scheduler`. In the future, it may be possible to
        change which scheduler is the default for a given cluster.

- Dealing with conflicts. All schedulers must use predicate functions that are at least as strict as
the ones that Kubelet applies when deciding whether to accept a pod, otherwise Kubelet and scheduler
may get into an infinite loop where Kubelet keeps rejecting a pod and scheduler keeps re-scheduling
it back the same node. To make it easier for people who write new schedulers to obey this rule, we will
create a library containing the predicates Kubelet uses. (See issue [#12744](https://github.com/kubernetes/kubernetes/issues/12744).)

In summary, in the initial version of this multi-scheduler design, we will achieve the following:

- If a pod has the annotation `scheduler.alpha.kubernetes.io/name: kube-scheduler` or the user does not explicitly
sets this annotation in the template, it will be picked up by default scheduler
- If the annotation is set and refers to a valid `scheduler-name`, it will be picked up by the scheduler of
specified `scheduler-name`
- If the annotation is set but refers to an invalid `scheduler-name`, the pod will not be picked by any scheduler.
The pod will keep PENDING.

### An example

```yaml
    kind: Pod
    apiVersion: v1
    metadata:
        name: pod-abc   
        labels:
            foo: bar
        annotations:
            scheduler.alpha.kubernetes.io/name: my-scheduler
```

This pod will be scheduled by "my-scheduler" and ignored by "kube-scheduler". If there is no running scheduler
of name "my-scheduler", the pod will never be scheduled.

## Next steps

1. Use admission controller to add and verify the annotation, and do some modification if necessary. For example, the
admission controller might add the scheduler annotation based on the namespace of the pod, and/or identify if
there are conflicting rules, and/or set a default value for the scheduler annotation, and/or reject pods on
which the client has set a scheduler annotation that does not correspond to a running scheduler.
2. Dynamic launching scheduler(s) and registering to admission controller (as an external call). This also
requires some work on authorization and authentication to control what schedulers can write the /binding
subresource of which pods.
3. Optimize the behaviors of priority functions in multi-scheduler scenario. In the case where multiple schedulers have
the same predicate and priority functions (for example, when using multiple schedulers for parallelism rather than to
customize the scheduling policies), all schedulers would tend to pick the same node as "best" when scheduling identical
pods and therefore would be likely to conflict on the Kubelet. To solve this problem, we can pass
an optional flag such as `--randomize-node-selection=N` to scheduler, setting this flag would cause the scheduler to pick
randomly among the top N nodes instead of the one with the highest score.

## Other issues/discussions related to scheduler design

- [#13580](https://github.com/kubernetes/kubernetes/pull/13580): scheduler extension
- [#17097](https://github.com/kubernetes/kubernetes/issues/17097): policy config file in pod template
- [#16845](https://github.com/kubernetes/kubernetes/issues/16845): scheduling groups of pods
- [#17208](https://github.com/kubernetes/kubernetes/issues/17208): guide to writing a new scheduler

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/multiple-schedulers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
