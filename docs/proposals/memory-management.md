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
[here](http://releases.k8s.io/release-1.0/docs/proposals/memory-management.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Memory Management in Kubernetes

**Author**: Vishnu Kannan ([@vishh](https://github.com/vishh))

**Status**: Implementation pending

#### Goals

1. Document existing state of memory management.

2. Explore opportunities for improving memory management in the future.

#### Why is memory management important?

Memory is a limited resource on machines. It is shared among multiple user containers and system processes. Unlike CPU, Memory is not compressible. Kubernetes being a cluster management system, aims to improve the utilization of memory on all nodes. Memory access latencies could vary a lot based on the architecture, state of the system, etc. When the available memory on a node becomes very low, the kernel can start killing processes. Kubernetes aims to provide a stable and predictable application performance by managing memory on the nodes, and providing efficient memory isolation for user containers.


#### How is memory managed as of now?

###### Requirements

* Cgroups are used to manage memory on the node.
* Kernel Out of Memory killer is expected to be enabled. [`/proc/<pid>/oom_score_adj`](https://www.kernel.org/doc/Documentation/filesystems/proc.txt) is used to manage priority for access to memory. The kernel assigns an `oom_score` to individual processes based on the memory usage of the process and a few [other policies](https://lwn.net/Articles/317814/). `oom_score_adj` is taken into account while calculating the oom_score. Higher values for `oom_score_adj` will increase the likelihood for being considered for OOM kills.

###### Node level memory management

Kubelet, the node agent, performs some amount of memory management. Specifically, it can

1. Expose the memory capacity and usage of the node.

2. Apply memory limits to user containers and individual processes on the node.

3. Set up cgroups to track the memory consumption of itself and other system processes like docker daemon, kube-proxy, sshd, etc. The cgroups being setup by kubelet are,
  a. `/docker-daemon`: The docker daemon is under this cgroup. This cgroup is limited to 70% of the total available memory on the node.
  b. `/system`: OS processes, excepting init, are under this cgroup.
  c. `/kubelet`: Kubelet is under this cgroup.
  d. `/kube-proxy`: The kube-proxy is under this cgroup.

4. Protect system processes from memory pressure on the node:
   The `oom_score_adj` for kubelet, kube-proxy and other system processes is set to `-999`
   The `oom_score_adj` for docker daemon is set to `-990`. Docker's memory usage has been unpredicatble in the past and docker daemons restarts are possible under system pressure.

5. Manage Quality of service for user containers: Three classes of containers are currently supported:

   a. `Guaranteed`: Memory is always guaranteed up to the request. These containers will not OOM unless they exceed their request, or system processes consume too much memory, or the kernel overhead is high and the memory in the node is being fully consumed. Note that `request` equals `limit` for this class of containers.

   b. `Burstable`: Memory is guaranteed up until the usage exceeds user specified `request`. These containers do not get killed as soon as they exceed their request. Kernel memory overhead and system processes memory usage will impact memory availability for this class of containers. They are more likely to be killed by the kernel OOM killer once they exceed their request.

   c. `Best-Effort`: Memory is never guaranteed. This class of containers are the first ones to be killed by the kernel whenever there is memory pressure on the node.

6. Detect OOM kills on containers: Docker tracks OOM kills on its containers.
Kubelet exposes this information via [ContainerStatus.State](../../pkg/api/v1/types.go#L1033), where the state is set to `ContainerStateTerminated` and `Reason` is set to `OOM Killed`.
Kubelet has a built-in exponential backoff mechanism that prevents OOMing containers from entering a crash loop and bringing down the node.


#### What are some of the known limitations?

1. The scheduler does not take into account the total usage on the node. Hence, memory can be easily over-committed. Whenever a job is scheduled without resource requests, the node becomes over-committed.

2. Swapping is turned off by default. Kubelet does not track or manage swap files or partitions.

3. A container in a pod with restart policy set to `Always` will be restarted continuously even if it OOMs. This is alleviated with the crash loop backoff feature. To improve node stability, that pod should be evicted and run on a different node.

4. Other than the Guaranteed class, containers in other classes don't have a memory limit.
A `Best-effort` container can use a lot of memory and result in high tail latencies for `Guaranteed` and `Burstable` class containers.
Memory allocations will hang until the kernel can free up memory, by identifying and killing a process in a container in a lower class.

5. Kubelet does not act on OOM kills.
A container can choose to continue running when one or more processes in the container were killed by the kernel OOM killer.
Because of this, the kubelet cannot deterministically associate container death with OOMs.
This affects higher level systems like auto-scaler which requires reliable OOM signals to update memory limits.

6. In the event of an OOM kill due to system memory pressure, the kubelet cannot identify the container of the killed process.

7. All burstable containers that exceed their request are equally likely to be killed. Ideally, the job that exceeds the most should be killed.

8. Memory usage of system daemons is unrestricted. This results in varying amount of memory availability on the node.


##### How to address these limitations?

We can improve memory management by making use of [kernel memcg features](https://www.kernel.org/doc/Documentation/cgroups/memory.txt), and by modifying the current system behavior. Some of these improvements are required in the short-term. The rest are either complex or require significant changes to the system.

###### What can be done right away?

1. Cleanup the entire container whenever a OOM kill is detected in a container.
Users should be able to override the default behavior for containers that have an in-built process manager that can handle OOM kills.

2. Set memory soft limits on Burstable containers to their request.
This should result in "Burstable" containers being pushed to their requested memory limit under system memory pressure.

3. Kubelet should expose `available` memory in addition to `capacity` and `usage`.
The scheduler can then schedule containers based on `available` instead of `capacity`, which is the case as of now.
To begin with `available` can be the difference between `capacity` and the total memory assigned to all system processes.
Since we do not restrict the resource usage of system daemons, the `available` resources can differ over time.
Another alternative is to create a shadow pod that will help statically reserve resources for system components.

###### What else can be done?

*Note: The features mentioned here are not recommended to be added to the system until the changes mentioned above are completed, and we have concrete evidence for further improving memory management in the system.*

1. Restrict resource usage of system processes.
Since the clusters are heterogeneous, the kubelet will have to come up with a limit for system processes dynamically, and apply those limits.
Kubernetes does not have full control over the node.
There is no restriction on what programs run on the node.
A Node Configuration Spec is required to partition node resources between components controlled by the kubelet and system processes that are run by the user.
A typical example here is a systemd machine where kubelet is run as a service, alonside all other system services. Kubelet does not understand the amount of resources that are dedicated for its pods and kubernetes system components.

1. Avoid overcommiting memory assigned to `Guaranteed` tasks by imposing limits on `Best-effort` and `Burstable` containers. By organizing memory cgroups hierarchically, `burstable` can be limited to use only the memory that has not been allocated for `Guaranteed` tasks. `Best-effort` containers can use the memory allocated to `Burstable` containers.

2. Provide pod level Out of Memory handling: In the event of an OOM kill of a process, in a container, in a pod, the entire pod could be stopped, marked as `Terminated`, and possibly re-scheduled.

3. Kubelet attempts to reduce the likelihood of system memory pressure:

   a. Kubelet can subscribe to memory pressure notifications from the kernel, and possibly evict low priority containers

   b. Kubelet can set thresholds based notifications on the root cgroup and evict containers that are most suitable candidates based on the QoS policy and current usage. Kubelet can try to limit total memory usage to under 90% of the memory capacity.

   This change requires defining the concept of priority and restart SLA for containers and pods in the system.

4. Disable OOM Kills on containers

   We can disable OOM killer for the individual containers. Kubelet (or another daemon) can watch for OOM kill notifications and evict containers based on QoS policy.

   However, disabling OOM kills at the system level will result in a system panic. If kubelet is unresponsive for any reason, we do not want the system to panic. Hence, disabling OOM killer at the system level is **not** recommended.

5. Enable Kernel Memory Accounting

   Enabling and restricting kernel memory usage will ensure that memory used by containers for kernel resources like Fed's, sockets, etc. will be restricted.

   This is a relatively new feature in the kernel. Until this feature prooves to be stable for some more time, this feature is not recommended to be enabled. Other caveats include, lack of page table accounting - random access to a huge file will result in a huge page table; some types of kernel memory (quota system) aren't evicted in favor of evicting text pages or killing a process under memory pressure.

#### Is that it?

This document is not complete. For example, this document does not explore Non-Uniform Memory Access and Swap management. Updates/Improvements to this documents are welcome!


#### References

* [QoS proposal](resource-qos.md)

* [Memory Cgroup](https://www.kernel.org/doc/Documentation/cgroups/memory.txt)

* [OOM Kill Policy](https://lwn.net/Articles/317814/)

* [RLimit cgroup](https://lwn.net/Articles/448435/)



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/memory-management.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
