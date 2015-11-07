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
[here](http://releases.k8s.io/release-1.0/docs/design/horizontal-pod-autoscaler.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Horizontal Pod Autoscaling

## Preface

This document briefly describes the design of the horizontal autoscaler for pods.
The autoscaler (implemented as a Kubernetes API resource and controller) is responsible for dynamically controlling
the number of replicas of some collection (e.g. the pods of a ReplicationController) to meet some objective(s),
for example a target per-pod CPU utilization.

This design supersedes [autoscaling.md](http://releases.k8s.io/release-1.0/docs/proposals/autoscaling.md).

## Overview

The resource usage of a serving application usually varies over time: sometimes the demand for the application rises,
and sometimes it drops.
In Kubernetes version 1.0, a user can only manually set the number of serving pods.
Our aim is to provide a mechanism for the automatic adjustment of the number of pods based on CPU utilization statistics
(a future version will allow autoscaling based on other resources/metrics).

## Scale Subresource

In Kubernetes version 1.1, we are introducing Scale subresource and implementing horizontal autoscaling of pods based on it.
Scale subresource is supported for replication controllers and deployments.
Scale subresource is a Virtual Resource (does not correspond to an object stored in etcd).
It is only present in the API as an interface that a controller (in this case the HorizontalPodAutoscaler) can use to dynamically scale
the number of replicas controlled by some other API object (currently ReplicationController and Deployment) and to learn the current number of replicas.
Scale is a subresource of the API object that it serves as the interface for.
The Scale subresource is useful because whenever we introduce another type we want to autoscale, we just need to implement the Scale subresource for it.
The wider discussion regarding Scale took place in [#1629](https://github.com/kubernetes/kubernetes/issues/1629).

Scale subresource is in API for replication controller or deployment under the following paths:

`apis/extensions/v1beta1/replicationcontrollers/myrc/scale`

`apis/extensions/v1beta1/deployments/mydeployment/scale`

It has the following structure:

```go
// represents a scaling request for a resource.
type Scale struct {
	unversioned.TypeMeta
	api.ObjectMeta

	// defines the behavior of the scale.
	Spec ScaleSpec

	// current status of the scale.
	Status ScaleStatus
}

// describes the attributes of a scale subresource
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	Replicas int `json:"replicas,omitempty"`
}

// represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int `json:"replicas"`

	// label query over pods that should match the replicas count.
	Selector map[string]string `json:"selector,omitempty"`
}
```

Writing to `ScaleSpec.Replicas` resizes the replication controller/deployment associated with
the given Scale subresource.
`ScaleStatus.Replicas` reports how many pods are currently running in the replication controller/deployment,
and `ScaleStatus.Selector` returns selector for the pods.

## HorizontalPodAutoscaler Object

In Kubernetes version 1.1, we are introducing HorizontalPodAutoscaler object. It is accessible under:

`apis/extensions/v1beta1/horizontalpodautoscalers/myautoscaler`

It has the following structure:

```go
// configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	unversioned.TypeMeta
	api.ObjectMeta

	// behavior of autoscaler.
	Spec HorizontalPodAutoscalerSpec

	// current information about the autoscaler.
	Status HorizontalPodAutoscalerStatus
}

// specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// reference to Scale subresource; horizontal pod autoscaler will learn the current resource
	// consumption from its status,and will set the desired number of pods by modifying its spec.
	ScaleRef SubresourceReference
	// lower limit for the number of pods that can be set by the autoscaler, default 1.
	MinReplicas *int
	// upper limit for the number of pods that can be set by the autoscaler.
	// It cannot be smaller than MinReplicas.
	MaxReplicas int
	// target average CPU utilization (represented as a percentage of requested CPU) over all the pods;
	// if not specified it defaults to the target CPU utilization at 80% of the requested resources.
	CPUUtilization *CPUTargetUtilization
}

type CPUTargetUtilization struct {
	// fraction of the requested CPU that should be utilized/used,
	// e.g. 70 means that 70% of the requested CPU should be in use.
	TargetPercentage int
}

// current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// most recent generation observed by this autoscaler.
	ObservedGeneration *int64

	// last time the HorizontalPodAutoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	LastScaleTime *unversioned.Time

	// current number of replicas of pods managed by this autoscaler.
	CurrentReplicas int

	// desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int

	// current average CPU utilization over all pods, represented as a percentage of requested CPU,
	// e.g. 70 means that an average pod is using now 70% of its requested CPU.
	CurrentCPUUtilizationPercentage *int
}
```

`ScaleRef` is a reference to the Scale subresource.
`MinReplicas`, `MaxReplicas` and `CPUUtilization` define autoscaler configuration.
We are also introducing HorizontalPodAutoscalerList object to enable listing all autoscalers in a namespace:

```go
// list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	unversioned.TypeMeta
	unversioned.ListMeta

	// list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler
}
```

## Autoscaling Algorithm

The autoscaler is implemented as a control loop. It periodically queries pods described by `Status.PodSelector` of Scale subresource, and collects their CPU utilization.
Then, it compares the arithmetic mean of the pods' CPU utilization with the target defined in `Spec.CPUUtilization`,
and adjust the replicas of the Scale if needed to match the target
(preserving condition: MinReplicas <= Replicas <= MaxReplicas).

The period of the autoscaler is controlled by `--horizontal-pod-autoscaler-sync-period` flag of controller manager.
The default value is 30 seconds.


CPU utilization is the recent CPU usage of a pod (average across the last 1 minute) divided by the CPU requested by the pod.
In Kubernetes version 1.1, CPU usage is taken directly from Heapster.
In future, there will be API on master for this purpose
(see [#11951](https://github.com/kubernetes/kubernetes/issues/11951)).

The target number of pods is calculated from the following formula:

```
TargetNumOfPods = ceil(sum(CurrentPodsCPUUtilization) / Target)
```

Starting and stopping pods may introduce noise to the metric (for instance, starting may temporarily increase CPU).
So, after each action, the autoscaler should wait some time for reliable data.
Scale-up can only happen if there was no rescaling within the last 3 minutes.
Scale-down will wait for 5 minutes from the last rescaling.
Moreover any scaling will only be made if: `avg(CurrentPodsConsumption) / Target` drops below 0.9 or increases above 1.1 (10% tolerance).
Such approach has two benefits:

* Autoscaler works in a conservative way.
  If new user load appears, it is important for us to rapidly increase the number of pods,
  so that user requests will not be rejected.
  Lowering the number of pods is not that urgent.

* Autoscaler avoids thrashing, i.e.: prevents rapid execution of conflicting decision if the load is not stable.

## Relative vs. absolute metrics

We chose values of the target metric to be relative (e.g. 90% of requested CPU resource) rather than absolute (e.g. 0.6 core) for the following reason.
If we choose absolute metric, user will need to guarantee that the target is lower than the request.
Otherwise, overloaded pods may not be able to consume more than the autoscaler's absolute target utilization,
thereby preventing the autoscaler from seeing high enough utilization to trigger it to scale up.
This may be especially troublesome when user changes requested resources for a pod
because they would need to also change the autoscaler utilization threshold.
Therefore, we decided to choose relative metric.
For user, it is enough to set it to a value smaller than 100%, and further changes of requested resources will not invalidate it.

## Support in kubectl

To make manipulation of HorizontalPodAutoscaler object simpler, we added support for
creating/updating/deleting/listing of HorizontalPodAutoscaler to kubectl.
In addition, in future, we are planning to add kubectl support for the following use-cases:
* When creating a replication controller or deployment with `kubectl create [-f]`, there should be
  a possibility to specify an additional autoscaler object.
  (This should work out-of-the-box when creation of autoscaler is supported by kubectl as we may include
  multiple objects in the same config file).
* *[future]* When running an image with `kubectl run`, there should be an additional option to create
  an autoscaler for it.
* *[future]* We will add a new command `kubectl autoscale` that will allow for easy creation of an autoscaler object
  for already existing replication controller/deployment.

## Next steps

We list here some features that are not supported in Kubernetes version 1.1.
However, we want to keep them in mind, as they will most probably be needed in future.
Our design is in general compatible with them.
*  *[future]* **Autoscale pods based on metrics different than CPU** (e.g. memory, network traffic, qps).
  This includes scaling based on a custom/application metric.
* *[future]* **Autoscale pods base on an aggregate metric.**
  Autoscaler, instead of computing average for a target metric across pods, will use a single, external, metric (e.g. qps metric from load balancer).
  The metric will be aggregated while the target will remain per-pod
  (e.g. when observing 100 qps on load balancer while the target is 20 qps per pod, autoscaler will set the number of replicas to 5).
* *[future]* **Autoscale pods based on multiple metrics.**
  If the target numbers of pods for different metrics are different, choose the largest target number of pods.
* *[future]* **Scale the number of pods starting from 0.**
  All pods can be turned-off, and then turned-on when there is a demand for them.
  When a request to service with no pods arrives, kube-proxy will generate an event for autoscaler
  to create a new pod.
  Discussed in [#3247](https://github.com/kubernetes/kubernetes/issues/3247).
* *[future]* **When scaling down, make more educated decision which pods to kill.**
  E.g.: if two or more pods from the same replication controller are on the same node, kill one of them.
  Discussed in [#4301](https://github.com/kubernetes/kubernetes/issues/4301).




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/horizontal-pod-autoscaler.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
