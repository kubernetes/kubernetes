# Table of Contents
  * [Accounting and Consuming Opaque Node-Level Integer Resources](#accounting-and-consuming-opaque-node-level-integer-resources)
    * [Terms](#terms)
      * [Opaque node-level resource](#opaque-node-level-resource)
    * [Requirements](#requirements)
    * [Concrete use cases](#concrete-use-cases)
    * [Goals](#goals)
    * [Non-goals](#non-goals)
    * [Design](#design)
      * [Denoting opaque integer resources](#denoting-opaque-integer-resources)
      * [Associating opaque integer resources with nodes](#associating-opaque-integer-resources-with-nodes)
      * [Updates to capacity/allocatable that invalidate bound pods.](#updates-to-capacityallocatable-that-invalidate-bound-pods)
        * [Scenario](#scenario)
        * [Options](#options)
      * [Ownership of NodeStatus updates](#ownership-of-nodestatus-updates)
    * [PoC implementation - detailed changes](#poc-implementation---detailed-changes)
      * [Changes to Kubelet](#changes-to-kubelet)
      * [Requesting opaque integer resources for a pod](#requesting-opaque-integer-resources-for-a-pod)
      * [Implementation](#implementation)
        * [Notable consequences](#notable-consequences)
    * [Related issues](#related-issues)
    * [Usage Example](#usage-example)
      * [Advertising Opaque Integer Resources](#advertising-opaque-integer-resources)
      * [Consuming Opaque Integer Resources](#consuming-opaque-integer-resources)

# Accounting and Consuming Opaque Node-Level Integer Resources

## Terms
### Opaque node-level resource
A resource associated with a node that has a consumable capacity, which is handled uniformly by the scheduler and Kubelet. (If there are specializations that hang off of the resource name, then it is not opaque!)

## Requirements
1. Associate available opaque integer resources with existing node(s).
2. Request opaque integer resources for a pod.
3. The scheduler must account opaque node-level integer resources similarly to existing first-class resources like logical CPU cores and main memory. Co-scheduled pods must not exceed the allocatable amount for any opaque integer resource.

## Concrete use cases
Even before implementing isolation, there is already user value in accounting opaque integer resources in the node.

* Exclusive access to discrete co-processors (GPGPUs, MICs, FPGAs, etc.)
* Counting slots allowed to access a shared parallel file system.
* Support for tracking additional cgroup quotas like `cpu_rt_runtime`.

## Goals
1. Associate opaque integer resources with nodes.
2. Request some portion of available capacity in the pod spec during pod creation.
3. Enable users to define and schedule on opaque integer resources in a vanilla Kubernetes cluster.

## Non-goals
1. Solve how opaque integer resource allocations are to be enforced/isolated.
2. Solve how opaque integer resources are discovered, beyond existing APIs.

## Design
### Denoting opaque integer resources
All opaque integer resources will share the same prefix so they can be handled properly in the system. [Proposal: `opaque-int.alpha.kubernetes.io/someResource`].

### Associating opaque integer resources with nodes
Upon discovery (however that happens), the node status can be PATCHed to update `Capacity` and `Allocatable` in `NodeStatus` to add a new resource. Everything that updates NodeStatus needs to use PATCH. The opaque integer resources of a node should be advertised in both `node.status.capacity` and `node.status.allocatable`.

### Updates to capacity/allocatable that invalidate bound pods.
#### Scenario
1. NodeStatus is patched to add an opaque integer resource to capacity and allocatable.
2. Pod(s) are scheduled on the node that consume some quantity of the opaque integer resource.
3. NodeStatus is patched again to reduce the capacity and allocatable for the opaque integer resource to less than has been allocated to the set of resident pods.

#### Options
1. Evict pods so that at most, the allocatable amount of opaque integer resource is consumed.
2. Disallow the NodeStatus update.
3. Do nothing.

### Ownership of NodeStatus updates
Kubelet assumes that it controls node status. After updating the NodeStatus object via the API server but before that Kubelet's next sync, the Kubelet's in-memory snapshot of NodeStatus would prevent correctly scheduled pods from running there.

Initially, it looks like there are two main approaches to resolving this. The first is to impose ordering such that NodeStatus updates are read by the Kubelet before validating any more pods. The second is to maintain the single-writer pattern by providing a way to tell Kubelet to update the NodeStatus.

Options:
1. When updating NodeStatus from the API server, we could also set some condition or taint that prevents pods from being scheduled there until it is cleared on the next sync.
2. Update the Kubelet first and let it advertise the new resources in NodeStatus when it syncs.
3. Since in Kubelet allocatable defaults to capacity on sync, and the scheduler looks at the Status.Allocatable field to make scheduling decisions, we could do the following.
	- Note: this doesn’t support the case where some opaque integer resource is being used outside of k8s, and that usage needs to be accounted in delta between allocatable and capacity. A simple workaround could be to only advertise the allocatable amount. That seems like a bit of a hack so maybe this option is only useful as the basis for a PoC.

	- A PoC implementation of this approach is here: https://github.com/intelsdi-x/kubernetes/pull/2

		- Advertise resource in `Status.Capacity` only using `PATCH`. 
		- On sync, Kubelet updates `Status.Allocatable` for the opaque integer resource, setting it equal to `Status.Capacity`. 
		- Scheduler is able to bind pods there. 

## PoC implementation - detailed changes 

### Changes to Kubelet
Edit the Kubelet node status update code to avoid overwriting advertised opaque integer resource capacity.

### Requesting opaque integer resources for a pod
We will use a regular ResourceRequirements (Limit and Request). These resources will initially be "alpha" and also be denoted with an opaque-int prefix as described above, The opaque integer prefix will cause the API server to skip validation. The scheduler will process it in new code that is added to handle such resources by matching up requests with available resources in the cluster.

### Implementation

Edit function PodFitsResources in the file `plugin/pkg/scheduler/algorithm/predicates/predicates.go` to iteratively compare requested opaque integer resources for the incoming pod with the allocatable resource quantity.

#### Notable consequences
- The Kubelet shares similar logic (actually it uses the same validation code). We need to be aware of potential failures that can arise due to the distributed nature of capacity with this proposal. Opaque resource discovery happens before scheduling any pod that requires that opaque integer resource.
- After opaque integer resource availability is advertised, users can require them in pods right away and the vanilla scheduler takes care of accounting and feasibility.
- Low computational overhead in scheduler (expected to be roughly linear in number of opaque integer resource types in request, and we are not expecting 100’s of opaque integer resources requested by each pod but rather 1’s. E.g., 1 FPGA/GPU, 1 Flash card, some queue here and there.) 

## Related issues
1. Accounting and feasibility for custom integer resources in the API server and scheduler. ([#28312][gh-issue-28312])
2. Create an opaque integer resource. ([#19082][gh-issue-19082]).

## Usage Example

### Advertising Opaque Integer Resources
Opaque integer resources are advertised by patching the node capcity via the api server. The following example shows an example script used for patching the node capacity via the api server. 

```sh
#!/bin/bash
curl --header "Content-Type:application/json-patch+json" \
        --request PATCH \
        --data '[{"op": "add", "path": "/status/capacity/opaque-int.alpha.kubernetes.io~1<resource-name>", "value": "<integer-value>"}]' \
        http://<apiserver-host>:<apiserver-port>/api/v1/nodes/<node-name>/status
```

where,

- `<resource-name>` is the name of the opaque integer resource being advertised. 
- `<integer-value>` is the number of the opaque integer resource available on the node. This value must be an integer. 
- `<apiserver-host>` is the hostname or the IP of the kube-apiserver. 
- `<apiserver-port>` is the port on which the kube-apiserver is listening. 
- `<node-name>` is the name of the node for which we want to update the capacity.

With appropriate changes, the above script will update the capacity of the node. In the next sync with the api server, the kubelet corresponding to the node will update its own `capacity` and `allocatable`. Subsequently, the scheduler on its next sync with the api server updates the value of allocatable resource in its own cache. 


### Consuming Opaque Integer Resources

After the advertisement of the opaque integer resource as explained above, it is ready to be consumed like a regular resource (e.g., CPU or memory). The following spec shows how to configure a pod to consume an opaque integer resource called `resource1`. Assuming that there are enough resources available to be allocated, the scheduler will bind this pod to the appropriate node in the Kubernetes cluster. 

```yaml 
apiVersion: v1
kind: Pod
metadata:
    name: demo-w-oir
spec:
  containers:
    - name: nginx
      image: nginx:1.10
      ports:
        - containerPort: 80
      resources:
        requests:
          memory: "32Mi"
          cpu: "250m"
          opaque-int.alpha.kubernetes.io/resource1: "4"
        limits:
          memory: "64Mi"
          cpu: "500m"
          opaque-int.alpha.kubernetes.io/resource1: "6"
 ```

[gh-issue-28312]: https://github.com/kubernetes/kubernetes/issues/28312 
[gh-issue-19082]: https://github.com/kubernetes/kubernetes/issues/19082
