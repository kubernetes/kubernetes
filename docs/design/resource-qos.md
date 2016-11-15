# Resource Quality of Service in Kubernetes

**Author(s)**: Vishnu Kannan (vishh@), Ananya Kumar (@AnanyaKumar)
**Last Updated**: 5/17/2016

**Status**: Implemented

*This document presents the design of resource quality of service for containers in Kubernetes, and describes use cases and implementation details.*

## Introduction

This document describes the way Kubernetes provides different levels of Quality of Service to pods depending on what they *request*.
Pods that need to stay up reliably can request guaranteed resources, while pods with less stringent requirements can use resources with weaker or no guarantee.

Specifically, for each resource, containers specify a request, which is the amount of that resource that the system will guarantee to the container, and a limit which is the maximum amount that the system will allow the container to use.
The system computes pod level requests and limits by summing up per-resource requests and limits across all containers.
When request == limit, the resources are guaranteed, and when request < limit, the pod is guaranteed the request but can opportunistically scavenge the difference between request and limit if they are not being used by other containers.
This allows Kubernetes to oversubscribe nodes, which increases utilization, while at the same time maintaining resource guarantees for the containers that need guarantees.
Borg increased utilization by about 20% when it started allowing use of such non-guaranteed resources, and we hope to see similar improvements in Kubernetes.

## Requests and Limits

For each resource, containers can specify a resource request and limit, `0 <= request <= `[`Node Allocatable`](../proposals/node-allocatable.md) & `request <= limit <= Infinity`.
If a pod is successfully scheduled, the container is guaranteed the amount of resources requested.
Scheduling is based on `requests` and not `limits`.
The pods and its containers will not be allowed to exceed the specified limit.
How the request and limit are enforced depends on whether the resource is [compressible or incompressible](resources.md).

### Compressible Resource Guarantees

- For now, we are only supporting CPU.
- Pods are guaranteed to get the amount of CPU they request, they may or may not get additional CPU time (depending on the other jobs running). This isn't fully guaranteed today because cpu isolation is at the container level. Pod level cgroups will be introduced soon to achieve this goal.
- Excess CPU resources will be distributed based on the amount of CPU requested. For example, suppose container A requests for 600 milli CPUs, and container B requests for 300 milli CPUs. Suppose that both containers are trying to use as much CPU as they can. Then the extra 10 milli CPUs will be distributed to A and B in a 2:1 ratio (implementation discussed in later sections).
- Pods will be throttled if they exceed their limit. If limit is unspecified, then the pods can use excess CPU when available.

### Incompressible Resource Guarantees

- For now, we are only supporting memory.
- Pods will get the amount of memory they request, if they exceed their memory request, they could be killed (if some other pod needs memory), but if pods consume less memory than requested, they will not be killed (except in cases where system tasks or daemons need more memory).
- When Pods use more memory than their limit, a process that is using the most amount of memory, inside one of the pod's containers, will be killed by the kernel.

### Admission/Scheduling Policy

- Pods will be admitted by Kubelet & scheduled by the scheduler based on the sum of requests of its containers. The scheduler & kubelet will ensure that sum of requests of all containers is within the node's [allocatable](../proposals/node-allocatable.md) capacity (for both memory and CPU).

## QoS Classes

In an overcommitted system (where sum of limits > machine capacity) containers might eventually have to be killed, for example if the system runs out of CPU or memory resources. Ideally, we should kill containers that are less important. For each resource, we divide containers into 3 QoS classes: *Guaranteed*, *Burstable*, and *Best-Effort*, in decreasing order of priority.

The relationship between "Requests and Limits" and "QoS Classes" is subtle. Theoretically, the policy of classifying pods into QoS classes is orthogonal to the requests and limits specified for the container. Hypothetically, users could use an (currently unplanned) API to specify whether a pod is guaranteed or best-effort. However, in the current design, the policy of classifying pods into QoS classes is intimately tied to "Requests and Limits" - in fact, QoS classes are used to implement some of the memory guarantees described in the previous section.

Pods can be of one of 3 different classes:

- If `limits` and optionally `requests` (not equal to `0`) are set for all resources across all containers and they are *equal*, then the container is classified as **Guaranteed**.

Examples:

```yaml
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
				memory: 100Mi
```

```yaml
containers:
	name: foo
		resources:
			limits:
				cpu: 10m
				memory: 1Gi
			requests:
				cpu: 10m
				memory: 1Gi

	name: bar
		resources:
			limits:
				cpu: 100m
				memory: 100Mi
			requests:
				cpu: 100m
				memory: 100Mi
```

- If `requests` and optionally `limits` are set (not equal to `0`) for one or more resources across one or more containers, and they are *not equal*, then the pod is classified as **Burstable**.
When `limits` are not specified, they default to the node capacity.

Examples:

Container `bar` has not resources specified.

```yaml
containers:
	name: foo
		resources:
			limits:
				cpu: 10m
				memory: 1Gi
			requests:
				cpu: 10m
				memory: 1Gi

	name: bar
```

Container `foo` and `bar` have limits set for different resources.

```yaml
containers:
	name: foo
		resources:
			limits:
				memory: 1Gi

	name: bar
		resources:
			limits:
				cpu: 100m
```

Container `foo` has no limits set, and `bar` has neither requests nor limits specified.

```yaml
containers:
	name: foo
		resources:
			requests:
				cpu: 10m
				memory: 1Gi

	name: bar
```

- If `requests` and `limits` are not set for all of the resources, across all containers, then the pod is classified as **Best-Effort**.

Examples:

```yaml
containers:
	name: foo
		resources:
	name: bar
		resources:
```

Pods will not be killed if CPU guarantees cannot be met (for example if system tasks or daemons take up lots of CPU), they will be temporarily throttled.

Memory is an incompressible resource and so let's discuss the semantics of memory management a bit.

- *Best-Effort* pods will be treated as lowest priority. Processes in these pods are the first to get killed if the system runs out of memory.
These containers can use any amount of free memory in the node though.

- *Guaranteed* pods are considered top-priority and are guaranteed to not be killed until they exceed their limits, or if the system is under memory pressure and there are no lower priority containers that can be evicted.

- *Burstable* pods have some form of minimal resource guarantee, but can use more resources when available.
Under system memory pressure, these containers are more likely to be killed once they exceed their requests and no *Best-Effort* pods exist.

### OOM Score configuration at the Nodes

Pod OOM score configuration
- Note that the OOM score of a process is 10 times the % of memory the process consumes, adjusted by OOM_SCORE_ADJ, barring exceptions (e.g. process is launched by root). Processes with higher OOM scores are killed.
- The base OOM score is between 0 and 1000, so if process A’s OOM_SCORE_ADJ - process B’s OOM_SCORE_ADJ is over a 1000, then process A will always be OOM killed before B.
- The final OOM score of a process is also between 0 and 1000

*Best-effort*
	- Set OOM_SCORE_ADJ: 1000
	- So processes in best-effort containers will have an OOM_SCORE of 1000

*Guaranteed*
	- Set OOM_SCORE_ADJ: -998
	- So processes in guaranteed containers will have an OOM_SCORE of 0 or 1

*Burstable*
	- If total memory request > 99.8% of available memory, OOM_SCORE_ADJ: 2
	- Otherwise, set OOM_SCORE_ADJ to 1000 - 10 * (% of memory requested)
	- This ensures that the OOM_SCORE of burstable pod is > 1
	- If memory request is `0`, OOM_SCORE_ADJ is set to `999`.
	- So burstable pods will be killed if they conflict with guaranteed pods
	- If a burstable pod uses less memory than requested, its OOM_SCORE < 1000
	- So best-effort pods will be killed if they conflict with burstable pods using less than requested memory
	- If a process in burstable pod's container uses more memory than what the container had requested, its OOM_SCORE will be 1000, if not its OOM_SCORE will be < 1000
	- Assuming that a container typically has a single big process, if a burstable pod's container that uses more memory than requested conflicts with another burstable pod's container using less memory than requested, the former will be killed
	- If burstable pod's containers with multiple processes conflict, then the formula for OOM scores is a heuristic, it will not ensure "Request and Limit" guarantees.

*Pod infra containers* or *Special Pod init process*
  - OOM_SCORE_ADJ: -998

*Kubelet, Docker*
  - OOM_SCORE_ADJ: -999 (won’t be OOM killed)
  - Hack, because these critical tasks might die if they conflict with guaranteed containers. In the future, we should place all user-pods into a separate cgroup, and set a limit on the memory they can consume.

## Known issues and possible improvements

The above implementation provides for basic oversubscription with protection, but there are a few known limitations.

#### Support for Swap

- The current QoS policy assumes that swap is disabled. If swap is enabled, then resource guarantees (for pods that specify resource requirements) will not hold. For example, suppose 2 guaranteed pods have reached their memory limit. They can continue allocating memory by utilizing disk space. Eventually, if there isn’t enough swap space, processes in the pods might get killed. The node must take into account swap space explicitly for providing deterministic isolation behavior.

## Alternative QoS Class Policy

An alternative is to have user-specified numerical priorities that guide Kubelet on which tasks to kill (if the node runs out of memory, lower priority tasks will be killed).
A strict hierarchy of user-specified numerical priorities is not desirable because:

1. Achieved behavior would be emergent based on how users assigned priorities to their pods. No particular SLO could be delivered by the system, and usage would be subject to gaming if not restricted administratively
2. Changes to desired priority bands would require changes to all user pod configurations.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/resource-qos.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
