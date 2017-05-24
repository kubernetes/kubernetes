# Pod level resource management in Kubelet

**Author**: Buddha Prakash (@dubstack), Vishnu Kannan (@vishh)

**Last Updated**: 06/23/2016

**Status**: Draft Proposal (WIP)

This document proposes a design for introducing pod level resource accounting to Kubernetes, and outlines the implementation and rollout plan.

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Pod level resource management in Kubelet](#pod-level-resource-management-in-kubelet)
  - [Introduction](#introduction)
  - [Non Goals](#non-goals)
  - [Motivations](#motivations)
  - [Design](#design)
    - [Proposed cgroup hierarchy:](#proposed-cgroup-hierarchy)
      - [QoS classes](#qos-classes)
      - [Guaranteed](#guaranteed)
      - [Burstable](#burstable)
      - [Best Effort](#best-effort)
    - [With Systemd](#with-systemd)
    - [Hierarchy Outline](#hierarchy-outline)
      - [QoS Policy Design Decisions](#qos-policy-design-decisions)
  - [Implementation Plan](#implementation-plan)
      - [Top level Cgroups for QoS tiers](#top-level-cgroups-for-qos-tiers)
      - [Pod level Cgroup creation and deletion (Docker runtime)](#pod-level-cgroup-creation-and-deletion-docker-runtime)
      - [Container level cgroups](#container-level-cgroups)
      - [Rkt runtime](#rkt-runtime)
      - [Add Pod level metrics to Kubelet's metrics provider](#add-pod-level-metrics-to-kubelets-metrics-provider)
  - [Rollout Plan](#rollout-plan)
  - [Implementation Status](#implementation-status)

<!-- END MUNGE: GENERATED_TOC -->

## Introduction

As of now [Quality of Service(QoS)](../../docs/design/resource-qos.md) is not enforced at a pod level. Excepting pod evictions, all the other QoS features are not applicable at the pod level.
To better support QoS, there is a need to add support for pod level resource accounting in Kubernetes.

We propose to have a unified cgroup hierarchy with pod level cgroups for better resource management. We will have a cgroup hierarchy with top level cgroups for the three QoS classes Guaranteed, Burstable and BestEffort. Pods (and their containers) belonging to a QoS class will be grouped under these top level QoS cgroups. And all containers in a pod are nested under the pod cgroup.

The proposed cgroup hierarchy would allow for more efficient resource management and lead to improvements in node reliability.
This would also allow for significant latency optimizations in terms of pod eviction on nodes with the use of pod level resource usage metrics.
This document provides a basic outline of how we plan to implement and rollout this feature.


## Non Goals

- Pod level disk accounting will not be tackled in this proposal.
- Pod level resource specification in the Kubernetes API will not be tackled in this proposal.

## Motivations

Kubernetes currently supports container level isolation only and lets users specify resource requests/limits on the containers [Compute Resources](../../docs/design/resources.md). The `kubelet` creates a cgroup sandbox (via it's container runtime) for each container.


There are a few shortcomings to the current model.
 - Existing QoS support does not apply to pods as a whole. On-going work to support pod level eviction using QoS requires all containers in a pod to belong to the same class. By having pod level cgroups, it is easy to track pod level usage and make eviction decisions.
 - Infrastructure overhead per pod is currently charged to the node. The overhead of setting up and managing the pod sandbox is currently accounted to the node. If the pod sandbox is a bit expensive, like in the case of hyper, having pod level accounting becomes critical.
 - For the docker runtime we have a containerd-shim which is a small library that sits in front of a runtime implementation allowing it to be reparented to init, handle reattach from the caller etc. With pod level cgroups containerd-shim can be charged to the pod instead of the machine.
 - If a container exits, all its anonymous pages (tmpfs) gets accounted to the machine (root). With pod level cgroups, that usage can also be attributed to the pod.
 - Let containers share resources - with pod level limits, a pod with a Burstable container and a BestEffort container is classified as Burstable pod. The BestEffort container is able to consume slack resources not used by the Burstable container, and still be capped by the overall pod level limits.

## Design

High level requirements for the design are as follows:
 - Do not break existing users. Ideally, there should be no changes to the Kubernetes API semantics.
 - Support multiple cgroup managers - systemd, cgroupfs, etc.

How we intend to achieve these high level goals is covered in greater detail in the Implementation Plan.

We use the following denotations in the sections below:

For the three QoS classes
`G⇒ Guaranteed QoS, Bu⇒ Burstable QoS, BE⇒ BestEffort QoS`

For the value specified for the --qos-memory-overcommitment flag
`qmo⇒ qos-memory-overcommitment`

Currently the Kubelet highly prioritizes resource utilization and thus allows BE pods to use as much resources as they want. And in case of OOM the BE pods are first to be killed. We follow this policy as G pods often don't use the amount of resource request they specify. By overcommiting the node the BE pods are able to utilize these left over resources. And in case of OOM the BE pods are evicted by the eviciton manager. But there is some latency involved in the pod eviction process which can be a cause of concern in latency-sensitive servers. On such servers we would want to avoid OOM conditions on the node. Pod level cgroups allow us to restrict the amount of available resources to the BE pods. So reserving the requested resources for the G and Bu pods would allow us to avoid invoking the OOM killer.


We add a flag `qos-memory-overcommitment` to kubelet which would allow users to configure the percentage of memory overcommitment on the node. We have the default as 100, so by default we allow complete overcommitment on the node and let the BE pod use as much memory as it wants, and not reserve any resources for the G and Bu pods. As expected if there is an OOM in such a case we first kill the BE pods before the G and Bu pods.
On the other hand if a user wants to ensure very predictable tail latency for latency-sensitive servers he would need to set qos-memory-overcommitment to a really low value(preferrably 0). In this case memory resources would be reserved for the G and Bu pods and BE pods would be able to use only the left over memory resource.

Examples in the next section.

### Proposed cgroup hierarchy:

For the initial implementation we will only support limits for cpu and memory resources.

#### QoS classes

A pod can belong to one of the following 3 QoS classes: Guaranteed, Burstable, and BestEffort, in decreasing order of priority.

#### Guaranteed

`G` pods will be placed at the `$Root` cgroup by default. `$Root` is the system root i.e. "/" by default and if `--cgroup-root` flag is used then we use the specified cgroup-root as the `$Root`. To ensure Kubelet's idempotent behaviour we follow a pod cgroup naming format which is opaque and deterministic. Say we have a pod with UID: `5f9b19c9-3a30-11e6-8eea-28d2444e470d` the pod cgroup PodUID would be named: `pod-5f9b19c93a3011e6-8eea28d2444e470d`.


__Note__: The cgroup-root flag would allow the user to configure the root of the QoS cgroup hierarchy. Hence cgroup-root would be redefined as the root of QoS cgroup hierarchy and not containers.

```
/PodUID/cpu.quota = cpu limit of Pod  
/PodUID/cpu.shares = cpu request of Pod  
/PodUID/memory.limit_in_bytes = memory limit of Pod
```

Example:
We have two pods Pod1 and Pod2 having Pod Spec given below

```yaml
kind: Pod
metadata:
    name: Pod1
spec:
    containers:
        name: foo
            resources:
                limits:
                    cpu: 10m
                    memory: 1Gi
        name: bar
            resources:
                limits:
                    cpu: 100m
                    memory: 2Gi
```

```yaml
kind: Pod
metadata:
    name: Pod2
spec:
    containers:
        name: foo
            resources:
                limits:
                    cpu: 20m
                    memory: 2Gii
```

Pod1 and Pod2 are both classified as `G` and are nested under the `Root` cgroup.

```
/Pod1/cpu.quota = 110m  
/Pod1/cpu.shares = 110m  
/Pod2/cpu.quota = 20m  
/Pod2/cpu.shares = 20m  
/Pod1/memory.limit_in_bytes = 3Gi  
/Pod2/memory.limit_in_bytes = 2Gi
```

#### Burstable

We have the following resource parameters for the `Bu` cgroup.

```
/Bu/cpu.shares = summation of cpu requests of all Bu pods  
/Bu/PodUID/cpu.quota = Pod Cpu Limit  
/Bu/PodUID/cpu.shares = Pod Cpu Request   
/Bu/memory.limit_in_bytes = Allocatable - {(summation of memory requests/limits of `G` pods)*(1-qom/100)}
/Bu/PodUID/memory.limit_in_bytes = Pod memory limit
```

`Note: For the `Bu` QoS when limits are not specified for any one of the containers, the Pod limit defaults to the node resource allocatable quantity.`

Example:
We have two pods Pod3 and Pod4 having Pod Spec given below:

```yaml
kind: Pod
metadata:
    name: Pod3
spec:
    containers:
        name: foo
            resources:
                limits:
                    cpu: 50m
                    memory: 2Gi
                requests:
                    cpu: 20m
                    memory: 1Gi
        name: bar
            resources:
                limits:
                    cpu: 100m
                    memory: 1Gi
```

```yaml
kind: Pod
metadata:
    name: Pod4
spec:
    containers:
        name: foo
            resources:
                limits:
                    cpu: 20m
                    memory: 2Gi
                requests:
                    cpu: 10m
                    memory: 1Gi  
```

Pod3 and Pod4 are both classified as `Bu` and are hence nested under the Bu cgroup
And for `qom` = 0

```
/Bu/cpu.shares = 30m  
/Bu/Pod3/cpu.quota = 150m  
/Bu/Pod3/cpu.shares = 20m  
/Bu/Pod4/cpu.quota = 20m  
/Bu/Pod4/cpu.shares = 10m  
/Bu/memory.limit_in_bytes = Allocatable - 5Gi  
/Bu/Pod3/memory.limit_in_bytes = 3Gi  
/Bu/Pod4/memory.limit_in_bytes = 2Gi  
```

#### Best Effort

For pods belonging to the `BE` QoS we don't set any quota.

```
/BE/cpu.shares = 2  
/BE/cpu.quota= not set  
/BE/memory.limit_in_bytes = Allocatable - {(summation of memory requests of all `G` and `Bu` pods)*(1-qom/100)}
/BE/PodUID/memory.limit_in_bytes = no limit  
```

Example:
We have a pod 'Pod5' having Pod Spec given below:

```yaml
kind: Pod
metadata:
    name: Pod5
spec:
    containers:
        name: foo
            resources:
        name: bar
            resources:
```

Pod5 is classified as `BE` and is hence nested under the BE cgroup
And for `qom` = 0

```
/BE/cpu.shares = 2  
/BE/cpu.quota= not set  
/BE/memory.limit_in_bytes = Allocatable - 7Gi  
/BE/Pod5/memory.limit_in_bytes = no limit  
```

### With Systemd

In systemd we have slices for the three top level QoS class. Further each pod is a subslice of exactly one of the three QoS slices. Each container in a pod belongs to a scope nested under the qosclass-pod slice.

Example:  We plan to have the following cgroup hierarchy on systemd systems

```
/memory/G-PodUID.slice/containerUID.scope
/cpu,cpuacct/G-PodUID.slice/containerUID.scope
/memory/Bu.slice/Bu-PodUID.slice/containerUID.scope
/cpu,cpuacct/Bu.slice/Bu-PodUID.slice/containerUID.scope
/memory/BE.slice/BE-PodUID.slice/containerUID.scope
/cpu,cpuacct/BE.slice/BE-PodUID.slice/containerUID.scope
```

### Hierarchy Outline

- "$Root" is the system root of the node i.e. "/" by default and if `--cgroup-root` is specified then the specified cgroup-root is used as "$Root".
- We have a top level QoS cgroup for the `Bu` and `BE` QoS classes.
- But we __dont__ have a separate cgroup for the `G` QoS class. `G` pod cgroups are brought up directly under the `Root` cgroup.
- Each pod has its own cgroup which is nested under the cgroup matching the pod's QoS class.
- All containers brought up by the pod are nested under the pod's cgroup.
- system-reserved cgroup contains the system specific processes.
- kube-reserved cgroup contains the kubelet specific daemons.

```
$ROOT
  |
  +- Pod1
  |   |
  |   +- Container1
  |   +- Container2
  |   ...
  +- Pod2
  |   +- Container3
  |   ...
  +- ...
  |
  +- Bu
  |   |
  |   +- Pod3
  |   |   |
  |   |   +- Container4
  |   |   ...
  |   +- Pod4
  |   |   +- Container5
  |   |   ...
  |   +- ...
  |
  +- BE
  |   |
  |   +- Pod5
  |   |   |
  |   |   +- Container6
  |   |   +- Container7
  |   |   ...
  |   +- ...
  |
  +- System-reserved
  |   |
  |   +- system
  |   +- docker (optional)
  |   +- ...
  |
  +- Kube-reserved 
  |   |
  |   +- kubelet
  |   +- docker (optional)
  |   +- ...
  |
```

#### QoS Policy Design Decisions

- This hierarchy highly prioritizes resource guarantees to the `G` over `Bu` and `BE` pods.
- By not having a separate cgroup for the `G` class, the hierarchy allows the `G` pods to burst and utilize all of Node's Allocatable capacity.
- The `BE` and `Bu` pods are strictly restricted from bursting and hogging resources and thus `G` Pods are guaranteed resource isolation.
- `BE` pods are treated as lowest priority. So for the `BE` QoS cgroup we set cpu shares to the lowest possible value ie.2. This ensures that the `BE` containers get a relatively small share of cpu time.
- Also we don't set any quota on the cpu resources as the containers on the `BE` pods can use any amount of free resources on the node.
- Having memory limit of `BE` cgroup as (Allocatable - summation of memory requests of `G` and `Bu` pods) would result in `BE` pods becoming more susceptible to being OOM killed. As more `G` and `Bu` pods are scheduled kubelet will more likely kill `BE` pods, even if the `G` and `Bu` pods are using less than their request since we will be dynamically reducing the size of `BE` m.limit_in_bytes. But this allows for better memory guarantees to the `G` and `Bu` pods.

## Implementation Plan

The implementation plan is outlined in the next sections.
We will have a 'experimental-cgroups-per-qos' flag to specify if the user wants to use the QoS based cgroup hierarchy. The flag would be set to false by default at least in v1.5.

#### Top level Cgroups for QoS tiers

Two top level cgroups for `Bu` and `BE` QoS classes are created when Kubelet starts to run on a node. All `G` pods cgroups are by default nested under the `Root`. So we dont create a top level cgroup for the `G` class. For raw cgroup systems we would use libcontainers cgroups manager for general cgroup management(cgroup creation/destruction). But for systemd we don't have equivalent support for slice management in libcontainer yet. So we will be adding support for the same in the Kubelet. These cgroups are only created once on Kubelet initialization as a part of node setup. Also on systemd these cgroups are transient units and will not survive reboot.

#### Pod level Cgroup creation and deletion (Docker runtime)

- When a new pod is brought up, its QoS class is firstly determined.
- We add an interface to Kubelet’s ContainerManager to create and delete pod level cgroups under the cgroup that matches the pod’s QoS class.
- This interface will be pluggable. Kubelet will support both systemd and raw cgroups based __cgroup__ drivers. We will be using the --cgroup-driver flag proposed in the [Systemd Node Spec](kubelet-systemd.md) to specify the cgroup driver.
- We inject creation and deletion of pod level cgroups into the pod workers.
- As new pods are added QoS class cgroup parameters are updated to match the resource requests by the Pod.

#### Container level cgroups

Have docker manager create container cgroups under pod level cgroups. With the docker runtime, we will pass --cgroup-parent using the syntax expected for the corresponding cgroup-driver the runtime was configured to use.

#### Rkt runtime

We want to have rkt create pods under a root QoS class that kubelet specifies, and set pod level cgroup parameters mentioned in this proposal by itself.

#### Add Pod level metrics to Kubelet's metrics provider

Update Kubelet’s metrics provider to include Pod level metrics. Use cAdvisor's cgroup subsystem information to determine various Pod level usage metrics.

`Note: Changes to cAdvisor might be necessary.`

## Rollout Plan

This feature will be opt-in in v1.4 and an opt-out in v1.5. We recommend users to drain their nodes and opt-in, before switching to v1.5, which will result in a no-op when v1.5 kubelet is rolled out.

## Implementation Status

The implementation goals of the first milestone are outlined below.
- [x] Finalize and submit Pod Resource Management proposal for the project #26751
- [x] Refactor qos package to be used globally throughout the codebase #27749 #28093
- [x] Add interfaces for CgroupManager and CgroupManagerImpl which implements the CgroupManager interface and creates, destroys/updates cgroups using the libcontainer cgroupfs driver. #27755 #28566
- [x] Inject top level QoS Cgroup creation in the Kubelet and add e2e tests to test that behaviour. #27853
- [x] Add PodContainerManagerImpl Create and Destroy methods which implements the respective PodContainerManager methods using a cgroupfs driver. #28017
- [x] Have docker manager create container cgroups under pod level cgroups. Inject creation and deletion of pod cgroups into the pod workers. Add e2e tests to test this behaviour. #29049
- [x] Add support for updating policy for the pod cgroups. Add e2e tests to test this behaviour. #29087
- [ ] Enabling 'cgroup-per-qos' flag in Kubelet: The user is expected to drain the node and restart it before enabling this feature, but as a fallback we also want to allow the user to just restart the kubelet with the cgroup-per-qos flag enabled to use this feature. As a part of this we need to figure out a policy for pods having Restart Policy: Never. More details in this [issue](https://github.com/kubernetes/kubernetes/issues/29946).
- [ ] Removing terminated pod's Cgroup : We need to cleanup the pod's cgroup once the pod is terminated. More details in this [issue](https://github.com/kubernetes/kubernetes/issues/29927).
- [ ] Kubelet needs to ensure that the cgroup settings are what the kubelet expects them to be. If security is not of concern, one can assume that once kubelet applies cgroups setting successfully, the values will never change unless kubelet changes it. If security is of concern, then kubelet will have to ensure that the cgroup values meet its requirements and then continue to watch for updates to cgroups via inotify and re-apply cgroup values if necessary.
Updating QoS limits needs to happen before pod cgroups values are updated. When pod cgroups are being deleted, QoS limits have to be updated after pod cgroup values have been updated for deletion or pod cgroups have been removed. Given that kubelet doesn't have any checkpoints and updates to QoS and pod cgroups are not atomic, kubelet needs to reconcile cgroups status whenever it restarts to ensure that the cgroups values match kubelet's expectation.
- [ ] [TEST] Opting in for this feature and rollbacks should be accompanied by detailed error message when killing pod intermittently.
- [ ] Add a systemd implementation for Cgroup Manager interface


Other smaller work items that we would be good to have before the release of this feature.
- [ ] Add Pod UID to the downward api which will help simplify the e2e testing logic.
- [ ] Check if parent cgroup exist and error out if they don’t.
- [ ] Set top level cgroup limit to resource allocatable until we support QoS level cgroup updates. If cgroup root is not `/` then set node resource allocatable as the cgroup resource limits on cgroup root.
- [ ] Add a NodeResourceAllocatableProvider which returns the amount of allocatable resources on the nodes. This interface would be used both by the Kubelet and ContainerManager.
- [ ] Add top level feasibility check to ensure that pod can be admitted on the node by estimating left over resources on the node.
- [ ] Log basic cgroup management ie. creation/deletion metrics


To better support our requirements we needed to make some changes/add features to Libcontainer as well

- [x] Allowing or denying all devices by writing 'a' to devices.allow or devices.deny is
not possible once the device cgroups has children. Libcontainer doesn’t have the option of skipping updates on parent devices cgroup. opencontainers/runc/pull/958
- [x] To use libcontainer for creating and managing cgroups in the Kubelet, I would like to just create a cgroup with no pid attached and if need be apply a pid to the cgroup later on. But libcontainer did not support cgroup creation without attaching a pid. opencontainers/runc/pull/956






<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/pod-resource-management.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
