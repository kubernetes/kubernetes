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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# DRAFT: Cluster capacity analysis

@ingvagabund

September 2016

## Introduction and definition

As new pods get scheduled on nodes in a cluster, more resources get consumed.
Monitoring available resources in the cluster is very important as operators can increase the current resources
in time before all of them get exhausted.
Or, carry different steps that lead to increase of available resources.

Cluster capacity consists of capacities of individual cluster nodes.
Capacity covers cpu, memory, disk space and other resources.

Goal is to analyse remaining allocatable resources and estimate available capacity that is still consumable.

## Motivation and use cases

Scheduler decides on which node a pod gets scheduled.
The decision depends on many factors such as available cpu, memory, disk, volumes, etc.
As long as enough resources is available, pods are scheduled.
On the other hand, when a pod can not be scheduled due to insufficient resources,
admin is usually made aware too late.
With knowledge of the current cluster capacity admin can be warned in advance.
Which can result in addition of new nodes (horizontal scaling) or increase of resources of existing nodes (vertical scaling) before any pod becomes pending.

With introduction of ``kubectl top`` command (https://github.com/kubernetes/kubernetes/pull/28844),
an admin can see the actual utilization (cpu, memory, storage) of individual nodes.
The command can answer question such as:

* How many cpu/memory for a given node is utilized
* How many pods are running and how many resources they consume

Allocated and overall resources of a node can be retrieved with `kubectl describe node` (look for `Capacity` and `Allocated resources`).
`Pod` section shows individual pods and their resource consumption.

Still, information about total amount of remaining capacity is not sufficient.
Cluster capacity consists of capacities of individual nodes and thus the entire resource space is fragmented.
As the Pod is the minimal schedulable unit, entire cluster space can be seen as a collection of pods.
Each pod corresponds to a quantum of resources causing the cluster capacity to be quantised.

Thus, in order to get more precise estimation of remaining capacity,
admin needs to be able to ask the cluster question "How many pods of given requirements can be scheduled".

Depending on whether an administrator is interested in capacity of overall cluster or a subset of nodes,
one needs to take into account:

* Allocated memory for all pods
* Allocated memory for each namespace
* Allocated CPU for all pods
* Allocated CPU for each namespace
* Allocated PV attachments on the nodes

[Compute resources](http://kubernetes.io/docs/user-guide/compute-resources/) document describes how to compute resources.
Resource usage of each pod is part of Pod status.
If a pod is scheduled without sufficient resources, scheduling fails with `PodExceedsFreeCPU` or `PodExceedsFreeMemory`.

Based on configuration of individual nodes, each node can report [disk and memory pressure](kubelet-eviction.md#node-conditions) for different percentual usage.
In order to take the pressure into account, node configuration could be reported to the Apiserver as part of the ``NodeInfo`` as well.
Or queried once the [dynamic Kubelet settings](https://github.com/kubernetes/kubernetes/pull/29459) is implemented.

### Pod requirements

There are various requirements a pod can specify.
As each pod consists of at least one container, administrator can set [resource limits and requests](http://kubernetes.io/docs/api-reference/v1/definitions/#_v1_resourcerequirements) for each container.
If set, container provides better information about consumed resources (e.g. memory, cpu).
Each pod can specify a list of volumes.
Some of the volumes can be claimed and provisioned with volume dynamic provisioner.
Some pods are meant to be run on dedicated nodes.

In general, requirements can include (the list can grow over time):

* cpu
* memory
* required disk space (a.k.a disk bytes)
* minimal disk IOPs
* gpu cores
* volumes
* [taints](../../docs/design/taint-toleration-dedicated.md)
* labels affinity & anti-affinity for [pods](../../docs/design/podaffinity.md) and [nodes](../../docs/design/nodeaffinity.md)
* opaque integer resources

Resources can be divided between first-class resources (memory, cpu, disk, etc.) and non-first class resources.

Other resource types can include (e.g. see [resource types](../../docs/design/resources.md#resource-types)):

* Network bandwidth (a.k.a NIC b/w)
* Network operations
* Network packets per second
* Storage space
* Storage time
* Storage operations
* Counted resources

### Use cases

**Use Case 1**: A namespace is configured to allow it to allocate 40GB of memory.
The POD with the largest amount of memory limit is 4GB.
The namespace has a node label selector for `region1, compute nodes`.
As an operator I need to know how many 4GB pods can be placed given the physical resources available
on the nodes that match the node selector of the namespace.
I also need to know how many of the largest pod definitions can be scaled up on the
namespace (in this case, the 4GB pod can be scaled up to 10,
assuming it's the only one defined).

**Use Case 2**: A namespace is configured to allow for 4 total CPUs to be used.
The POD with the largest amount of CPU allocated is given half a CPU.
The namespace has a node label selector for `region2, compute nodes`.
As an operator I need to know how many half CPU pods can be placed given the physical resources available
on the nodes that match the node selector of the namespace.
I also need to know how many of the largest pod definitions can be scaled up
on this namespace (in this case, the half CPU pod can be scaled up to 8,
assuming it's the only one defined).

**Use Case 3**: A POD definition has a PV claim.
As an operator I need to know how many nodes have available attachments such that
the POD can be scheduled to nodes that fit the namespace node selector.

### Autoscaling

Introduction of new nodes or increase of node resources is not the only way how to deal with insufficient cluster capacity.
One can take advantage of [Horizontal Pod Autoscaling](http://kubernetes.io/docs/user-guide/horizontal-pod-autoscaling/) scaling a number of replicas (Replication Controller, Deployment, ReplicaSet) based on the current resource utilization.
Or [Cluster Autoscaler](https://github.com/mwielgus/contrib/blob/7262a5c1ad19abb8a495ad0cb5c5b340d4230d0e/cluster-autoscaler/README.md).
Still, the autoscaling is another indicator of insufficient resources which can result in node scaling.

## Design considerations and design space

Assuming the following:

* predicting remaining cluster capacity on pod bases (e.g. "How many pods of a given shape I can schedule")
* seeing the scheduling algorithms as blackboxes, using the same configuration
* prediction must not change the state of the cluster
* provide general framework usable for any scheduling algorithm

Scheduling process uses two sets of functions: predicates and priority functions.
Predicates select all nodes suitable for scheduling.
Priority functions choose the most suitable node from the selected nodes.
Both predicates and priority functions may need to query the current state of the cluster during execution.
Queries can include node's free cpu and memory, volume claims, annotations, etc.
Once a pod gets scheduled on a node, it can change its free resources and change the decision basis of the next scheduling iteration.

Thus, when designing the predictor, one has to take into account the current cluster state and compute its next without changing the current one.
I.e. simulate the pod scheduling and binding. The simulation consists of (not exclusively in this order):

* estimation of the current cluster state (node info, available volume claims, etc.)
* execution of one scheduling iteration
* update of scheduler caches based on scheduled pod

Some consumed resources do not have to be specified in the pod (e.g. volume claims).
These kind of consumed resources must be simulated accordingly.

Once the current state is captured, prediction is no longer dependent on the cluster state.
Thus, the estimation does not have to correspond to the real scheduling since:

* some nodes can be deleted, some nodes can be evacuated, some nodes can be re-labeled/re-annotated
* some pods can get evicted in the time of analysis
* remaining capacity of nodes can increase/decrease

Scheduler does not have to be aware of all node constraints.
For instance, Kubelet admission controller can have more knowledge not reported in node status.

### Future aspects to consider

One could take into account [shared resources](http://kubernetes.io/docs/user-guide/compute-resources/#planned-improvements) when predicting free capacity (with fragmentation) as well.
Currently, limit and requests are supported for cpu and memory resource types only.
In future, the types can be extended to node disk space resources and custom [resource types](../../docs/design/resources.md#resource-types).

Another thing to keep in mind is overcommitment with multiple levels of QoS.
Or consider compressible vs. incompressible resources.

#### Cluster resource subdivision

Allow cluster resources to be subdivided (https://github.com/kubernetes/kubernetes/issues/442).

#### Resource quota

[Resource quota](http://kubernetes.io/docs/admin/resourcequota/) can influence number of pods that can be scheduled [per namespace](http://kubernetes.io/docs/admin/resourcequota/#object-count-quota) as well.
Though, this limitation is artificial and independent of limitation of the underlying system (e.g. amount of cores or memory of each node,
maximal number of GCE persistent disks per cloud), it can be taken into account in future implementations.

Example: In a cluster with a capacity of 32 GiB RAM, and 16 cores, let team A use 20 Gib and 10 cores, let B use 10GiB and 4 cores, and hold 2GiB and 2 cores in reserve for future allocation.

#### Federated scheduling

Out of scope of the document.
Can be implemented as an aggregation of cluster capacities of individual sub-clusters.

#### Multiple schedulers

Different workloads may require different schedulers.
For that reason, Kubernetes allows to specify [multiple schedulers](https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/multiple-schedulers.md) and annotate each pod with one of the schedulers.

#### Scheduler extensions

Scheduler can extend its predicates/priority functions by delegating to [external processes](https://github.com/kubernetes/kubernetes/blob/master/docs/design/scheduler_extender.md).
As the extensions are part of a scheduling algorithm, there is no need to consider them.

## Implementation

Goals:

* Build the implementation over the existing code.
* Run the final implementation as a self-standing application in a pod (under `kube-system` namespace).
* Provide REST-API to query questions like "approximately how many more pods could i schedule with this shape?"
* Integrate the cluster capacity into ``kubectl`` (e.g. ``kubectl cluster-capacity``)

### Code analysis

Currently, each iteration of the default scheduler implementation consists of the following [steps](../../plugin/pkg/scheduler/scheduler.go#L93):

1. ask for the next pod ``s.config.NextPod``
1. schedule the pod ``s.config.Algorithm.Schedule``
1. update metrics ``metrics.SchedulingAlgorithmLatency.Observe``
1. update scheduler cache ``s.config.SchedulerCache.AssumePod``
1. pod binding ``s.config.Binder.Bind``

Scheduling itself (``Schedule``) consists of the following [steps](../../plugin/pkg/scheduler/generic_scheduler.go#L79):

1. retrieve a list of nodes ``nodeLister.List``
1. update node info ``g.cache.UpdateNodeNameToInfoMap``
1. filter nodes with predicates ``findNodesThatFit``
1. prioritized remaining nodes ``PrioritizeNodes``
1. pick the most suitable node ``g.selectHost``

Scheduler keeps a number of caches which capture the current state of the cluster.
Pod cache is continuously updated [outside](../../plugin/pkg/scheduler/factory/factory.go#L128) the scheduler.
The same holds for the [list of nodes](../../plugin/pkg/scheduler/factory/factory.go#L140).

As the individual predicates and priority functions need to query other objects such as services, controllers, etc.,
other caches are populated as well:

* PV cache
* PVC cache
* Service cache
* Controller cache
* ReplicaSet cache

All caches are continuously updated via [reflectors](../../plugin/pkg/scheduler/factory/factory.go#L387).

Once the reflectors are destroyed, no cache is updated anymore.

Each iteration of the scheduler calls ``s.config.NextPod()`` function which pops one pod at a time from the queue of unscheduled pods.
The queue is again continuously updated via [reflector](../../plugin/pkg/scheduler/factory/factory.go#L389).

Once the scheduler decides what node to schedule a pod on, the pod is bind to the Apiserver.
Once the pod is scheduled on a node, the Kubelet runs the pod and updates node info that is periodically sent to the Apiserver.
The node info is then reflected in the scheduler cache and the process is repeated.

### Prediction

As all the caches are outside of any scheduling algorithm, the prediction is scheduling algorithm independent.
Thus, based on the configuration, administrator can use various algorithms while keeping the same predictive framework.
The only requirements for any scheduling algorithm is to implement ``ScheduleAlgorithm`` [interface](../../plugin/pkg/scheduler/algorithm/scheduler_interface.go).

As all the caches are populated via reflectors and informers, the current state can be captured the same way as is done in the scheduler factory.
Once populated, all reflectors and informers are stopped since the predication is done independently of the cluster state.

At this point the prediction starts:

1. schedule a pod (with provided scheduling algorithm)
1. simulate deployment of the pod through the Kubelet (i.e. compute all resources consumed by the pod, including volume claims, etc.)
1. recompute status of the cluster stored in caches
1. update caches

#### Scheduler

For purposes of the prediction we need light-weighted version of the [scheduler](../../plugin/pkg/scheduler/scheduler.go).
No need for metrics, event recorder or pod updater.
The binding can hide the actual computation of consumed resources.

```Go
type Scheduler struct {
	config *Config
}

type Config struct {
	// It is expected that changes made via SchedulerCache will be observed
	// by NodeLister and Algorithm.
	SchedulerCache schedulercache.Cache
	NodeLister     algorithm.NodeLister
	Algorithm      algorithm.ScheduleAlgorithm
	Binder         Binder

	// NextPod always returns a pod, never blocking as the pod is taken
	// from a predefined sequence of pods that is repeated forever
	NextPod func() *api.Pod
}

// New returns a new scheduler.
func New(c *Config) *Scheduler {
	s := &Scheduler{
		config: c,
	}
	return s
}

// Run begins watching and scheduling. It starts a goroutine and returns immediately.
func (s *Scheduler) Run() int {
	pods_scheduled := 0

	while(s.scheduleOne()) {
		pods_scheduled++
	}

	return pods_scheduled
}

func (s *Scheduler) scheduleOne() bool {
	pod := s.config.NextPod()

	glog.V(3).Infof("Attempting to predict pod scheduling: %v/%v", pod.Namespace, pod.Name)
	dest, err := s.config.Algorithm.Schedule(pod, s.config.NodeLister)
	if err != nil {
		glog.V(1).Infof("Failed to schedule pod: %v/%v, due to: %q", pod.Namespace, pod.Name, err)
		return false
	}

	b := &api.Binding{
		ObjectMeta: api.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name},
		Target: api.ObjectReference{
			Kind: "Node",
			Name: dest,
		},
	}

	// Binding consists of recomputation of cached resources and objects
	err := s.config.Binder.Bind(b)
	if err != nil {
		glog.V(1).Infof("Failed to simulated pod binding: %v/%v, due to %q", pod.Namespace, pod.Name, err)
		return false
	}

	return true
}
```

It is possible to use the same scheduler and use empty event recorder and metrics.
[Pod updater](../../plugin/pkg/scheduler/factory/factory.go#L593) delegates requests to client.

#### Binding and local client

When a pod is scheduled, scheduler's caches are populated with the current state of the cluster.
The population is done through client querying the Apiserver.
Once a pod is scheduled it is bind back to the Apiserver.
Kubelet acknowledge the pod and makes sure all containers specified in the pod are running.
Kubelet reflects the new pod and reports the state back to the Apiserver.
As scheduler's caches are continuously updated (through functions implementing ``Watch`` and ``List`` interface),
node statuses (consumed resources, volume claims, etc.) are updated and the process starts over.

As the prediction works with the caches captured before prediction,
there is no communication with the Apiserver nor with the Kubelet.
Thus, the client implementing the same API has to be introduced.
Aim of the client is to provide:

* lister and watcher for each affected object so the caches can be continuously updated the same way
* recomputation of the cluster state simulating the Kubelet's behaviour (e.g. increasing node's cpu or memory)

At the same time, all operations are carried over local caches (different from the scheduler ones).
The binding process delegates pod to [client](../../plugin/pkg/scheduler/factory/factory.go#L336).
Thus, it is simulated as a part of the recomputation.

As the Kubelet uses scheduler's [NodeInfo.addPod](../../plugin/pkg/scheduler/schedulercache/node_info.go#L171) to compute the overall consumption of resources, the recomputation can reuse the same code.

Before the prediction starts, all the predictor caches need to be populated.
By default, every scheduler configuration is built from the [config factory](../../plugin/pkg/scheduler/factory/factory.go#L100).
The factory provides various ways how to build the configuration.
Building process also [initialize](../../plugin/pkg/scheduler/factory/factory.go#L387) reflectors and informers responsible for caches population and updates.
Each reflector and informer has its own cache and use exactly one of the following functions to create ListerWatcher:

* `createUnassignedNonTerminatedPodLW`
* `createAssignedNonTerminatedPodLW`
* `createNodeLW`
* `createPersistentVolumeLW`
* `createPersistentVolumeClaimLW`
* `createServiceLW`
* `createControllerLW`
* `createReplicaSetLW`

Thus, when implementing the local client, all the caches need to be populated with the current state
using exactly the same ListerWatchers per predictor cache.
Once the caches are populated, all the reflectors and informers are terminated.
Thus, the local client can return content of the caches to simulate communication with the Apiserver.

At this point the factory can build the configuration pointing to the local client.
Scheduler gets created and the prediction can start.
At the end of each scheduling iteration the binding process is converted to cache update.
Normally, each binding results in watch event once the pod gets run on kubelet and node status gets reflected in the Apiserver.
Thus, as a part of updating the predictor caches, watch event is generated (e.g. in the [RestClient](../../pkg/watch/streamwatcher.go#L114) it corresponds to sending an item to a channel).

From the point of view of the scheduler the local client behaves the same way as any client,
i.e. it provides ``Listers``/``Watchers`` implementing ``List`` and ``Watch`` interface

## Examples

**Expected output** (still very sci-fi, ``kubectl cluster-capacity`` is only illustrative, the command can change):

```
$ kubectl cluster-capacity -f pod.json
PodRequirements:
- cpu: 2.5
- memory: 40Mi
- PVClaim:
  - count: 2
  - total_storage: 10Gi
- labels:
  - name: zone
    value: Europe
  - name: type
    value: HPC

The cluster can schedule 23 instance(s) of the pod.
```

Optionally, list of nodes individual pods get scheduled on can be returned.
Or report a reason which prevented the next pod from being scheduled.

Additionally, it may be useful to put boundaries on the number of scheduling iterations depending on cluster size or performance cost.
In some situation it is sufficient to simulate scheduling of ``N`` instances of the pod and stop.

## Roadmap

1. implement the local client, prediction framework, REST API, limit considered resources for cpu, memory and gpu
1. extend considered resources, extend prediction

## Future

* Take re-scheduler into account with possible de-fragmentation

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/cluster-capacity.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
