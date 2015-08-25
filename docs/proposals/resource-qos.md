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
[here](http://releases.k8s.io/release-1.0/docs/proposals/resource-qos.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Resource Quality of Service in Kubernetes

**Author**: Ananya Kumar (@AnanyaKumar)

**Status**: Draft proposal; prototype in progress.

*This document presents the design of resource quality of service for containers in Kubernetes, and describes use cases and implementation details.*

## Motivation

Kubernetes allocates resources to containers in a simple way. Users can specify resource limits for containers. For example, a user can specify a 1gb memory limit for a container. The scheduler uses resource limits to schedule containers (technically, the scheduler schedules pods comprised of containers). For example, the scheduler will not place 5 containers with a 1gb memory limit onto a machine with 4gb memory. Currently, Kubernetes does not have robust mechanisms to ensure that containers run reliably on an overcommitted system.

In the current implementation, **if users specify limits for every container, cluster utilization is poor**. Containers often don’t use all the resources that they request which leads to a lot of wasted resources. For example, we might have 4 containers, each reserving 1GB of memory in a node with 4GB memory but only using 500MB of memory. Theoretically, we could fit more containers on the node, but Kubernetes will not schedule new pods (with specified limits) on the node.

A possible solution is to launch containers without specified limits - containers that don't ask for any resource guarantees. But **containers with limits specified are not very well protected from containers without limits specified**. If a container without a specified memory limit goes overboard and uses lots of memory, other containers (with specified memory limits) might be killed. This is bad, because users often want a way to launch containers that have resources guarantees, and that stay up reliably.

This proposal provides mechanisms for oversubscribing nodes while maintaining resource guarantees, by allowing containers to specify levels of resource guarantees. Containers will be able to *request* for a minimum resource guarantee. The *request* is different from the *limit* - containers will not be allowed to exceed resource limits. With this change, users can launch *best-effort* containers with 0 request. Best-effort containers use resources only if not being used by other containers, and can be used for resource-scavenging. Supporting best-effort containers in Borg increased utilization by about 20%, and we hope to see similar improvements in Kubernetes.

## Requests and Limits

Note: this section describes the functionality that QoS should eventually provide. Due to implementation issues, providing some of these guarantees, while maintaining our broader goals of efficient cluster utilization, is difficult. Later sections will go into the nuances of how the functionality will be achieved, and limitations of the initial implementation.

For each resource, containers can specify a resource request and limit, 0 <= request <= limit <= Infinity. If the container is successfully scheduled, the container is guaranteed the amount of resource requested. The container will not be allowed to exceed the specified limit. How the request and limit are enforced depends on whether the resource is [compressible or incompressible](../../docs/design/resources.md).

### Compressible Resource Guarantees

- For now, we are only supporting CPU.
- Containers are guaranteed to get the amount of CPU they request, they may or may not get additional CPU time (depending on the other jobs running).
- Excess CPU resources will be distributed based on the amount of CPU requested. For example, suppose container A requests for 60% of the CPU, and container B requests for 30% of the CPU. Suppose that both containers are trying to use as much CPU as they can. Then the extra 10% of CPU will be distributed to A and B in a 2:1 ratio (implementation discussed in later sections).
- Containers will be throttled if they exceed their limit. If limit is unspecified, then the containers can use excess CPU when available.

### Incompressible Resource Guarantees

- For now, we are only supporting memory.
- Containers will get the amount of memory they request, if they exceed their memory request, they could be killed (if some other container needs memory), but if containers consume fewer resources than requested, they will not be killed (except in cases where system tasks or daemons need more memory).
- Containers will be killed if they use more memory than their limit.

### Kubelet Admission Policy

- Pods will be admitted by Kubelet based on the sum of requests of its containers. The Kubelet will ensure that sum of requests of all containers (over all pods) is within the system’s resource (for both memory and CPU).

## QoS Classes

In an overcommitted system (where sum of requests > machine capacity) containers might eventually have to be killed, for example if the system runs out of CPU or memory resources. Ideally, we should kill containers that are less important. For each resource, we divide containers into 3 QoS classes: *Guaranteed*, *Burstable*, and *Best-Effort*, in decreasing order of priority.

The relationship between "Requests and Limits" and "QoS Classes" is subtle. Theoretically, the policy of classifying containers into QoS classes is orthogonal to the requests and limits specified for the container. Hypothetically, users could use an (currently unplanned) API to specify whether a container is guaranteed or best-effort. However, in this proposal, the policy of classifying containers into QoS classes is intimately tied to "Requests and Limits" - in fact, QoS classes are used to implement some of the memory guarantees described in the previous section.

For each resource, containers will be split into 3 different classes
- For now, we will only focus on memory. Containers will not be killed if CPU guarantees cannot be met (for example if system tasks or daemons take up lots of CPU), they will be temporarily throttled.
- Containers with a 0 memory request are classified as memory *Best-Effort*. These containers are not requesting resource guarantees, and will be treated as lowest priority (processes in these containers are the first to get killed if the system runs out of memory).
- Containers with the same request and limit and non-zero request are classified as memory *Guaranteed*. These containers ask for a well-defined amount of the resource and are considered top-priority (with respect to memory usage).
- All other containers are memory *Burstable* - middle priority containers that have some form of minimal resource guarantee, but can use more resources when available.
- In the current policy and implementation, best-effort containers are technically a subset of Burstable containers (where the request is 0), but they are a very important special case. Memory best-effort containers don't ask for any resource guarantees so they can utilize unused resources in a cluster (resource scavenging).

### Alternative QoS Class Policy

An alternative is to have user-specified numerical priorities that guide Kubelet on which tasks to kill (if the node runs out of memory, lower priority tasks will be killed). A strict hierarchy of user-specified numerical priorities is not desirable because:

1. Achieved behavior would be emergent based on how users assigned priorities to their containers. No particular SLO could be delivered by the system, and usage would be subject to gaming if not restricted administratively
2. Changes to desired priority bands would require changes to all user container configurations.

## Implementation

### To implement requests (PR #12035):

API changes for request
- Default request to limit, if limit is specified but request is not (api/v1/defaults.go)
- Add validation code that checks request <= limit, and validation test cases (api/validation/validation.go)

Scheduler Changes
- Use requests instead of limits in CheckPodsExceedingCapacity and PodFitsResources (scheduler/algorithm/predicates.go)

Container Manager Changes
- Use requests to assign CPU shares for Docker (kubelet/dockertools/container_manager.go)
- RKT changes will be implemented in a later iteration

### QoS Classes (PR #12182):

For now, we will be implementing QoS classes using OOM scores. However, system OOM kills are expensive, and without kernel modifications we cannot rely on system OOM kills to enforce burstable class guarantees. Eventually, we will need to layer control loops on top of OOM score assignment.

Add kubelet/qos/policy.go
- Decides which memory QoS class a container is in (based on the policy described above)
- Decides what OOM score all processes in a container should get

Change memory overcommit mode
- Right now overcommit mode is off on the machines we set up, so if there isn’t enough memory malloc will return null. This prevents QoS, because best-effort containers won’t be killed. Instead, when there isn’t enough memory, and guaranteed containers call malloc, they may not get the memory they want. We want memory guaranteed containers to get the memory they request, and force out memory best-effort containers.
- Change the memory overcommit mode to 1, so that using excess memory starts the OOM killer. The implication is that malloc won't return null, a process will be killed instead.

Container OOM score configuration
- We’re focusing on Docker in this implementation (not RKT)
- OOM scores
  - Note that the OOM score of a process is 10 times the % of memory the process consumes, adjusted by OOM_SCORE_ADJ, barring exceptions (e.g. process is launched by root). Processes with higher OOM scores are killed.
  - The base OOM score is between 0 and 1000, so if process A’s OOM_SCORE_ADJ - process B’s OOM_SCORE_ADJ is over a 1000, then process A will always be OOM killed before B.
  - The final OOM score of a process is also between 0 and 1000
- Memory best-effort
  - Set OOM_SCORE_ADJ: 1000
  - So processes in best-effort containers will have an OOM_SCORE of 1000
- Memory guaranteed
  - Set OOM_SCORE_ADJ: -999
  - So processes in guaranteed containers will have an OOM_SCORE of 0 or 1
- Memory burstable
  - If total memory request > 99.8% of available memory, OOM_SCORE_ADJ: 2
  - Otherwise, set OOM_SCORE_ADJ to 1000 - 10 * (% of memory requested)
    - This ensures that the OOM_SCORE of burstable containers is > 1
    - So burstable containers will be killed if they conflict with guaranteed containers
    - If a burstable container uses less memory than requested, its OOM_SCORE < 1000
    - So best-effort containers will be killed if they conflict with burstable containers using less than requested memory
    - If a process in a burstable container uses more memory than the container requested, its OOM_SCORE will be 1000, if not its OOM_SCORE will be < 1000
    - Assuming that a container typically has a single big process, if a burstable container that uses more memory than requested conflicts with a burstable container using less memory than requested, the former will be killed
    - If burstable containers with multiple processes conflict, then the formula for OOM scores is a heuristic, it will not ensure "Request and Limit" guarantees. This is one reason why control loops will be added in subsequent iterations.
- Pod infrastructure container
  - OOM_SCORE_ADJ: -999
- Kubelet, Docker, Kube-Proxy
  - OOM_SCORE_ADJ: -999 (won’t be OOM killed)
  - Hack, because these critical tasks might die if they conflict with guaranteed containers. in the future, we should place all user-pods into a separate cgroup, and set a limit on the memory they can consume.

Setting OOM_SCORE_ADJ for a container
- Refactor existing ApplyOomScoreAdj to util/oom.go
- To set OOM_SCORE_ADJ of a container, we loop through all processes in the container, and set OOM_SCORE_ADJ
- We keep looping until the list of processes in the container stabilizes. This is sufficient because child processes inherit OOM_SCORE_ADJ.

## Implementation Issues and Extensions

The above implementation provides for basic oversubscription with protection, but there are a number of issues. Below is a list of issues and TODOs for each of them. The first iteration of QoS will not solve these problems, but we aim to solve them in subsequent iterations of QoS. This list is not exhaustive. We expect to add issues to the list, and reference issues and PRs associated with items on this list.

Supporting other platforms:
- **RKT**: The proposal focuses on Docker. TODO: add support for RKT.
- **Systemd**: Systemd platforms need to be handled in a different way. Handling distributions of Linux based on systemd is critical, because major Linux distributions like Debian and Ubuntu are moving to systemd. TODO: Add code to handle systemd based operating systems.

Protecting containers and guarantees:
- **Control loops**: The OOM score assignment is not perfect for burstable containers, and system OOM kills are expensive. TODO: Add a control loop to reduce memory pressure, while ensuring guarantees for various containers.
- **Kubelet, Kube-proxy, Docker daemon protection**: If a system is overcommitted with memory guaranteed containers, then all prcoesses will have an OOM_SCORE of 0. So Docker daemon could be killed instead of a container or pod being killed. TODO: Place all user-pods into a separate cgroup, and set a limit on the memory they can consume. Initially, the limits can be based on estimated memory usage of Kubelet, Kube-proxy, and CPU limits, eventually we can monitor the resources they consume.
- **OOM Assignment Races**: We cannot set OOM_SCORE_ADJ of a process until it has launched. This could lead to races. For example, suppose that a memory burstable container is using 70% of the system’s memory, and another burstable container is using 30% of the system’s memory. A best-effort burstable container attempts to launch on the Kubelet. Initially the best-effort container is using 2% of memory, and has an OOM_SCORE_ADJ of 20. So its OOM_SCORE is lower than the burstable  pod using 70% of system memory. The burstable pod will be evicted by the best-effort pod. Short-term TODO: Implement a restart policy where best-effort pods are immediately evicted if OOM killed, but burstable pods are given a few retries. Long-term TODO: push support for OOM scores in cgroups to the upstream Linux kernel.
- **Swap Memory**: The QoS proposal assumes that swap memory is disabled. If swap is enabled, then resource guarantees (for pods that specify resource requirements) will not hold. For example, suppose 2 guaranteed pods have reached their memory limit. They can start allocating memory on swap space. Eventually, if there isn’t enough swap space, processes in the pods might get killed. TODO: ensure that swap space is disabled on our cluster setups scripts.

Killing and eviction mechanics:
- **Killing Containers**:  Usually, containers cannot function properly if one of the constituent processes in the container is killed. TODO: When a process in a container is out of resource killed (e.g. OOM killed), kill the entire container.
- **Out of Resource Eviction**: If a container in a multi-container pod fails, we might want restart the entire pod instead of just restarting the container. In some cases (e.g. if a memory best-effort container is out of resource killed), we might change pods to "failed" phase and pods might need to be evicted. TODO: Draft a policy for out of resource eviction and implement it.

Maintaining CPU performance:
- **CPU-sharing Issues** Suppose that a node is running 2 container: a container A requesting for 50% of CPU (but without a CPU limit), and a container B not requesting for resoruces. Suppose that both pods try to use as much CPU as possible. After the proposal is implemented, A will get 100% of the CPU, and B will get around 0% of the CPU. However, a fairer scheme would give the Burstable container 75% of the CPU and the Best-Effort container 25% of the CPU (since resources past the Burstable container’s request are not guaranteed). TODO: think about whether this issue to be solved, implement a solution.
- **CPU kills**: System tasks or daemons like the Kubelet could consume more CPU, and we won't be able to guarantee containers the CPU amount they requested. If the situation persists, we might want to kill the container. TODO: Draft a policy for CPU usage killing and implement it.
- **CPU limits**: Enabling CPU limits can be problematic, because processes might be hard capped and might stall for a while. TODO: Enable CPU limits intelligently using CPU quota and core allocation.

Documentation:
- **QoS Class Status**: TODO: Add code to ContainerStatus in the API, so that it shows which memory and CPU classes a container is in.
- **Documentation**: TODO: add user docs for resource QoS

## Demo and Tests

Possible demos/E2E tests:
- Launch a couple of memory guaranteed containers on a node. Barrage the node with memory best-effort containers. The memory guaranteed containers should survive the onslaught of memory best-effort containers.
- Fill up a node with memory best-effort containers. Barrage the node with memory guaranteed containers. All memory best-effort containers should be evicted. This is a hard test, because the Kubelet, Kube-proxy, etc need to be well protected.
- Launch a container with 0 CPU request. The container, when run in isolation, should get to use the entire CPU. Then add a container with non-zero request that tries to use up CPU. The 0-requst containers should be throttled, and given a small number of CPU shares.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/resource-qos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
