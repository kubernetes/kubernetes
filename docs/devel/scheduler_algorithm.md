<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/scheduler_algorithm.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Scheduler Algorithm in Kubernetes

For each unscheduled Pod, the Kubernetes scheduler tries to find a node across the cluster according to a set of rules. A general introduction to the Kubernetes scheduler can be found at [scheduler.md](scheduler.md). In this document, the algorithm of how to select a node for the Pod is explained. There are two steps before a destination node of a Pod is chosen. The first step is filtering all the nodes and the second is ranking the remaining nodes to find a best fit for the Pod.

## Filtering the nodes

The purpose of filtering the nodes is to filter out the nodes that do not meet certain requirements of the Pod. For example, if the free resource on a node (measured by the capacity minus the sum of the resource requests of all the Pods that already run on the node) is less than the Pod's required resource, the node should not be considered in the ranking phase so it is filtered out. Currently, there are several "predicates" implementing different filtering policies, including:

- `NoDiskConflict`: Evaluate if a pod can fit due to the volumes it requests, and those that are already mounted.
- `NoVolumeZoneConflict`: Evaluate if the volumes a pod requests are available on the node, given the Zone restrictions.
- `PodFitsResources`: Check if the free resource (CPU and Memory) meets the requirement of the Pod. The free resource is measured by the capacity minus the sum of requests of all Pods on the node. To learn more about the resource QoS in Kubernetes, please check [QoS proposal](../design/resource-qos.md).
- `PodFitsHostPorts`: Check if any HostPort required by the Pod is already occupied on the node.
- `HostName`: Filter out all nodes except the one specified in the PodSpec's NodeName field.
- `MatchNodeSelector`: Check if the labels of the node match the labels specified in the Pod's `nodeSelector` field and, as of Kubernetes v1.2, also match the `scheduler.alpha.kubernetes.io/affinity` pod annotation if present. See [here](../user-guide/node-selection/) for more details on both.
- `MaxEBSVolumeCount`: Ensure that the number of attached ElasticBlockStore volumes does not exceed a maximum value (by default, 39, since Amazon recommends a maximum of 40 with one of those 40 reserved for the root volume -- see [Amazon's documentation](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/volume_limits.html#linux-specific-volume-limits)).  The maximum value can be controlled by setting the `KUBE_MAX_PD_VOLS` environment variable.
- `MaxGCEPDVolumeCount`: Ensure that the number of attached GCE PersistentDisk volumes does not exceed a maximum value (by default, 16, which is the maximum GCE allows -- see [GCE's documentation](https://cloud.google.com/compute/docs/disks/persistent-disks#limits_for_predefined_machine_types)).  The maximum value can be controlled by setting the `KUBE_MAX_PD_VOLS` environment variable.
- `CheckNodeMemoryPressure`: Check if a pod can be scheduled on a node reporting memory pressure condition. Currently, no ``BestEffort`` should be placed on a node under memory pressure as it gets automatically evicted by kubelet.
- `CheckNodeDiskPressure`: Check if a pod can be scheduled on a node reporting disk pressure condition. Currently, no pods should be placed on a node under disk pressure as it gets automatically evicted by kubelet.

The details of the above predicates can be found in [plugin/pkg/scheduler/algorithm/predicates/predicates.go](http://releases.k8s.io/HEAD/plugin/pkg/scheduler/algorithm/predicates/predicates.go). All predicates mentioned above can be used in combination to perform a sophisticated filtering policy. Kubernetes uses some, but not all, of these predicates by default. You can see which ones are used by default in [plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go](http://releases.k8s.io/HEAD/plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go).

## Ranking the nodes

The filtered nodes are considered suitable to host the Pod, and it is often that there are more than one nodes remaining. Kubernetes prioritizes the remaining nodes to find the "best" one for the Pod. The prioritization is performed by a set of priority functions. For each remaining node, a priority function gives a score which scales from 0-10 with 10 representing for "most preferred" and 0 for "least preferred". Each priority function is weighted by a positive number and the final score of each node is calculated by adding up all the weighted scores. For example, suppose there are two priority functions, `priorityFunc1` and `priorityFunc2` with weighting factors `weight1` and `weight2` respectively, the final score of some NodeA is:

    finalScoreNodeA = (weight1 * priorityFunc1) + (weight2 * priorityFunc2)

After the scores of all nodes are calculated, the node with highest score is chosen as the host of the Pod. If there are more than one nodes with equal highest scores, a random one among them is chosen.

Currently, Kubernetes scheduler provides some practical priority functions, including:

- `LeastRequestedPriority`: The node is prioritized based on the fraction of the node that would be free if the new Pod were scheduled onto the node. (In other words, (capacity - sum of requests of all Pods already on the node - request of Pod that is being scheduled) / capacity). CPU and memory are equally weighted. The node with the highest free fraction is the most preferred. Note that this priority function has the effect of spreading Pods across the nodes with respect to resource consumption.
- `BalancedResourceAllocation`: This priority function tries to put the Pod on a node such that the CPU and Memory utilization rate is balanced after the Pod is deployed.
- `SelectorSpreadPriority`: Spread Pods by minimizing the number of Pods belonging to the same service, replication controller, or replica set on the same node.  If zone information is present on the nodes, the priority will be adjusted so that pods are spread across zones and nodes.
- `CalculateAntiAffinityPriority`: Spread Pods by minimizing the number of Pods belonging to the same service on nodes with the same value for a particular label.
- `ImageLocalityPriority`: Nodes are prioritized based on locality of images requested by a pod. Nodes with larger size of already-installed packages required by the pod will be preferred over nodes with no already-installed packages required by the pod or a small total size of already-installed packages required by the pod.
- `NodeAffinityPriority`: (Kubernetes v1.2) Implements `preferredDuringSchedulingIgnoredDuringExecution` node affinity; see [here](../user-guide/node-selection/) for more details.

The details of the above priority functions can be found in [plugin/pkg/scheduler/algorithm/priorities](http://releases.k8s.io/HEAD/plugin/pkg/scheduler/algorithm/priorities/). Kubernetes uses some, but not all, of these priority functions by default. You can see which ones are used by default in [plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go](http://releases.k8s.io/HEAD/plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go). Similar as predicates, you can combine the above priority functions and assign weight factors (positive number) to them as you want (check [scheduler.md](scheduler.md) for how to customize).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/scheduler_algorithm.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
