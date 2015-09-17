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
[here](http://releases.k8s.io/release-1.0/docs/proposals/horizontal-pod-autoscaler.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Horizontal Pod Autoscaling

**Author**: Jerzy Szczepkowski (@jszczepkowski)

## Preface

This document briefly describes the design of the horizontal autoscaler for pods.
The autoscaler (implemented as a kubernetes control loop) will be responsible for automatically
choosing and setting the number of pods of a given type that run in a kubernetes cluster.

This proposal supersedes [autoscaling.md](http://releases.k8s.io/release-1.0/docs/proposals/autoscaling.md).

## Overview

The usage of a serving application usually vary over time: sometimes the demand for the application rises,
and sometimes it drops.
In Kubernetes version 1.0, a user can only manually set the number of serving pods.
Our aim is to provide a mechanism for the automatic adjustment of the number of pods based on usage statistics.

## Scale Subresource

We are going to introduce Scale subresource and implement horizontal autoscaling of pods based on it.
Scale subresource will be supported for replication controllers and deployments.
Scale subresource will be a Virtual Resource (will not be stored in etcd as a separate object).
It will be only present in API as an interface to accessing replication controller or deployment,
and the values of Scale fields will be inferred from the corresponding replication controller/deployment object.
HorizontalPodAutoscaler object will be bound with exactly one Scale subresource and will be
autoscaling associated replication controller/deployment through it.
The main advantage of such approach is that whenever we introduce another type we want to auto-scale,
we just need to implement Scale subresource for it (w/o modifying autoscaler code or API).
The wider discussion regarding Scale took place in [#1629](https://github.com/kubernetes/kubernetes/issues/1629).

Scale subresource will be present in API for replication controller or deployment under the following paths:

```api/vX/replicationcontrollers/myrc/scale```

```api/vX/deployments/mydeployment/scale```

It will have the following structure:

```go
// Scale subresource, applicable to ReplicationControllers and (in future) Deployment.
type Scale struct {
	api.TypeMeta
	api.ObjectMeta

	// Spec defines the behavior of the scale.
	Spec ScaleSpec

	// Status represents the current status of the scale.
	Status ScaleStatus
}

// ScaleSpec describes the attributes a Scale subresource
type ScaleSpec struct {
	// Replicas is the number of desired replicas.
	Replicas int
}

// ScaleStatus represents the current status of a Scale subresource.
type ScaleStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int

	// Selector is a label query over pods that should match the replicas count.
	Selector map[string]string
}

```

Writing ```ScaleSpec.Replicas``` will resize the replication controller/deployment associated with
the given Scale subresource.
```ScaleStatus.Replicas``` will report how many pods are currently running in the replication controller/deployment,
and ```ScaleStatus.Selector``` will return selector for the pods.

## HorizontalPodAutoscaler Object

We will introduce HorizontalPodAutoscaler object, it will be accessible under:

```
api/vX/horizontalpodautoscalers/myautoscaler
```

It will have the following structure:

```go
// HorizontalPodAutoscaler represents the configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	api.TypeMeta
	api.ObjectMeta

	// Spec defines the behaviour of autoscaler.
	Spec HorizontalPodAutoscalerSpec

	// Status represents the current information about the autoscaler.
	Status HorizontalPodAutoscalerStatus
}

// HorizontalPodAutoscalerSpec is the specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// ScaleRef is a reference to Scale subresource. HorizontalPodAutoscaler will learn the current
	// resource consumption from its status, and will set the desired number of pods by modifying its spec.
	ScaleRef *SubresourceReference
	// MinReplicas is the lower limit for the number of pods that can be set by the autoscaler.
	MinReplicas int
	// MaxReplicas is the upper limit for the number of pods that can be set by the autoscaler.
	// It cannot be smaller than MinReplicas.
	MaxReplicas int
	// Target is the target average consumption of the given resource that the autoscaler will try
	// to maintain by adjusting the desired number of pods.
	// Currently this can be either "cpu" or "memory".
	Target ResourceConsumption
}

// HorizontalPodAutoscalerStatus contains the current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// CurrentReplicas is the number of replicas of pods managed by this autoscaler.
	CurrentReplicas int

	// DesiredReplicas is the desired number of replicas of pods managed by this autoscaler.
	// The number may be different because pod downscaling is sometimes delayed to keep the number
	// of pods stable.
	DesiredReplicas int

	// CurrentConsumption is the current average consumption of the given resource that the autoscaler will
	// try to maintain by adjusting the desired number of pods.
	// Two types of resources are supported: "cpu" and "memory".
	CurrentConsumption ResourceConsumption

	// LastScaleTimestamp is the last time the HorizontalPodAutoscaler scaled the number of pods.
	// This is used by the autoscaler to control how often the number of pods is changed.
	LastScaleTimestamp *unversioned.Time
}

// ResourceConsumption is an object for specifying average resource consumption of a particular resource.
type ResourceConsumption struct {
	Resource api.ResourceName
	Quantity resource.Quantity
}
```

```Scale``` will be a reference to the Scale subresource.
```MinReplicas```, ```MaxReplicas``` and ```Target``` will define autoscaler configuration.
We will also introduce HorizontalPodAutoscalerList object to enable listing all autoscalers in the cluster:

```go
// HorizontalPodAutoscaler is a collection of pod autoscalers.
type HorizontalPodAutoscalerList struct {
	api.TypeMeta
	api.ListMeta

	Items []HorizontalPodAutoscaler
}
```

## Autoscaling Algorithm

The autoscaler will be implemented as a control loop.
It will periodically (e.g.: every 1 minute) query pods described by ```Status.PodSelector``` of Scale subresource,
and check their average CPU or memory usage from the last 1 minute
(there will be API on master for this purpose, see
[#11951](https://github.com/kubernetes/kubernetes/issues/11951).
Then, it will compare the current CPU or memory consumption with the Target,
and adjust the replicas of the Scale if needed to match the target
(preserving condition: MinReplicas <= Replicas <= MaxReplicas).

The target number of pods will be calculated from the following formula:

```
TargetNumOfPods =ceil(sum(CurrentPodsConsumption) / Target)
```

Starting and stopping pods may introduce noise to the metrics (for instance starting may temporarily increase
CPU and decrease average memory consumption) so, after each action, the autoscaler should wait some time for reliable data.

Scale-up will happen if there was no rescaling within the last 3 minutes.
Scale-down will wait for 10 minutes from the last rescaling. Moreover any scaling will only be made if

```
avg(CurrentPodsConsumption) / Target
```

drops below 0.9 or increases above 1.1 (10% tolerance). Such approach has two benefits:

* Autoscaler works in a conservative way.
  If new user load appears, it is important for us to rapidly increase the number of pods,
  so that user requests will not be rejected.
  Lowering the number of pods is not that urgent.

* Autoscaler avoids thrashing, i.e.: prevents rapid execution of conflicting decision if the load is not stable.

## Relative vs. absolute metrics

The question arises whether the values of the target metrics should be absolute (e.g.: 0.6 core, 100MB of RAM)
or relative (e.g.: 110% of resource request, 90% of resource limit).
The argument for the relative metrics is that when user changes resources for a pod,
she will not have to change the definition of the autoscaler object, as the relative metric will still be valid.
However, we want to be able to base autoscaling on custom metrics in the future.
Such metrics will rather be absolute (e.g.: the number of queries-per-second).
Therefore, we decided to give absolute values for the target metrics in the initial version.

Please note that when custom metrics are supported, it will be possible to create additional metrics
in heapster that will divide CPU/memory consumption by resource request/limit.
From autoscaler point of view the metrics will be absolute,
although such metrics will be bring the benefits of relative metrics to the user.


## Support in kubectl

To make manipulation on HorizontalPodAutoscaler object simpler, we will add support for
creating/updating/deletion/listing of HorizontalPodAutoscaler to kubectl.
In addition, we will add kubectl support for the following use-cases:
* When running an image with ```kubectl run```, there should be an additional option to create
  an autoscaler for it.
* When creating a replication controller or deployment with ```kubectl create [-f]```, there should be
  a possibility to specify an additional autoscaler object.
  (This should work out-of-the-box when creation of autoscaler is supported by kubectl as we may include
  multiple objects in the same config file).
* We will and a new command ```kubectl autoscale``` that will allow for easy creation of an autoscaler object
  for already existing replication controller/deployment.

## Next steps

We list here some features that will not be supported in the initial version of autoscaler.
However, we want to keep them in mind, as they will most probably be needed in future.
Our design is in general compatible with them.
* Autoscale pods based on metrics different than CPU & memory (e.g.: network traffic, qps).
  This includes scaling based on a custom metric.
* Autoscale pods based on multiple metrics.
  If the target numbers of pods for different metrics are different, choose the largest target number of pods.
* Scale the number of pods starting from 0: all pods can be turned-off,
  and then turned-on when there is a demand for them.
  When a request to service with no pods arrives, kube-proxy will generate an event for autoscaler
  to create a new pod.
  Discussed in [#3247](https://github.com/kubernetes/kubernetes/issues/3247).
* When scaling down, make more educated decision which pods to kill (e.g.: if two or more pods are on the same node, kill one of them).
  Discussed in [#4301](https://github.com/kubernetes/kubernetes/issues/4301).
* Allow rule based autoscaling: instead of specifying the target value for metric,
  specify a rule, e.g.: “if average CPU consumption of pod is higher than 80% add two more replicas”.
  This approach was initially suggested in
  [autoscaling.md](http://releases.k8s.io/release-1.0/docs/proposals/autoscaling.md) proposal.
  Before doing this, we need to evaluate why the target based scaling described in this proposal is not sufficient.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/horizontal-pod-autoscaler.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
