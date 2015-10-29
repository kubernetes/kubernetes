<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Scheduler Algorithm in Kubernetes

For each unscheduled Pod, the Kubernetes scheduler tries to find a node across the cluster according to a set of rules. A general introduction to the Kubernetes scheduler can be found at [scheduler.md](scheduler.md). In this document, the algorithm of how to select a node for the Pod is explained. There are two steps before a destination node of a Pod is chosen. The first step is filtering all the nodes and the second is ranking the remaining nodes to find a best fit for the Pod.

## Filtering the nodes

The purpose of filtering the nodes is to filter out the nodes that do not meet certain requirements of the Pod. For example, if the free resource on a node (measured by the capacity minus the sum of the resource requests of all the Pods that already run on the node) is less than the Pod's required resource, the node should not be considered in the ranking phase so it is filtered out. Currently, there are several "predicates" implementing different filtering policies, including:

- `NoDiskConflict`: Evaluate if a pod can fit due to the volumes it requests, and those that are already mounted.
- `PodFitsResources`: Check if the free resource (CPU and Memory) meets the requirement of the Pod. The free resource is measured by the capacity minus the sum of requests of all Pods on the node. To learn more about the resource QoS in Kubernetes, please check [QoS proposal](../proposals/resource-qos.md).
- `PodFitsHostPorts`: Check if any HostPort required by the Pod is already occupied on the node.
- `PodFitsHost`: Filter out all nodes except the one specified in the PodSpec's NodeName field.
- `PodSelectorMatches`: Check if the labels of the node match the labels specified in the Pod's `nodeSelector` field ([Here](../user-guide/node-selection/) is an example of how to use `nodeSelector` field).
- `CheckNodeLabelPresence`: Check if all the specified labels exist on a node or not, regardless of the value.

The details of the above predicates can be found in [plugin/pkg/scheduler/algorithm/predicates/predicates.go](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/algorithm/predicates/predicates.go). All predicates mentioned above can be used in combination to perform a sophisticated filtering policy. Kubernetes uses some, but not all, of these predicates by default. You can see which ones are used by default in [plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go).

## Ranking the nodes

The filtered nodes are considered suitable to host the Pod, and it is often that there are more than one nodes remaining. Kubernetes prioritizes the remaining nodes to find the "best" one for the Pod. The prioritization is performed by a set of priority functions. For each remaining node, a priority function gives a score which scales from 0-10 with 10 representing for "most preferred" and 0 for "least preferred". Each priority function is weighted by a positive number and the final score of each node is calculated by adding up all the weighted scores. For example, suppose there are two priority functions, `priorityFunc1` and `priorityFunc2` with weighting factors `weight1` and `weight2` respectively, the final score of some NodeA is:

    finalScoreNodeA = (weight1 * priorityFunc1) + (weight2 * priorityFunc2)

After the scores of all nodes are calculated, the node with highest score is chosen as the host of the Pod. If there are more than one nodes with equal highest scores, a random one among them is chosen.

Currently, Kubernetes scheduler provides some practical priority functions, including:

- `LeastRequestedPriority`: The node is prioritized based on the fraction of the node that would be free if the new Pod were scheduled onto the node. (In other words, (capacity - sum of requests of all Pods already on the node - request of Pod that is being scheduled) / capacity). CPU and memory are equally weighted. The node with the highest free fraction is the most preferred. Note that this priority function has the effect of spreading Pods across the nodes with respect to resource consumption.
- `CalculateNodeLabelPriority`: Prefer nodes that have the specified label.
- `BalancedResourceAllocation`: This priority function tries to put the Pod on a node such that the CPU and Memory utilization rate is balanced after the Pod is deployed.
- `CalculateSpreadPriority`: Spread Pods by minimizing the number of Pods belonging to the same service on the same node.
- `CalculateAntiAffinityPriority`: Spread Pods by minimizing the number of Pods belonging to the same service on nodes with the same value for a particular label.

The details of the above priority functions can be found in [plugin/pkg/scheduler/algorithm/priorities](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/algorithm/priorities/). Kubernetes uses some, but not all, of these priority functions by default. You can see which ones are used by default in [plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go](http://releases.k8s.io/release-1.1/plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go). Similar as predicates, you can combine the above priority functions and assign weight factors (positive number) to them as you want (check [scheduler.md](scheduler.md) for how to customize).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/scheduler_algorithm.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
