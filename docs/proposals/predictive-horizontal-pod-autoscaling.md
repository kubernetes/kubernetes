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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Predictive Horizontal Pod Autoscaling

**Author**: Matt McNaughton (@mattjmcnaughton)

## Preface

This document describes the addition of prediction to the
[horizontal pod autoscaler] (../user-guide/horizontal-pod-autoscaler.md).
This addition will decrease the amount of time pods spend operating outside of
their target value for resource consumption.

## Overview and Motivation

In 1.1, Kubernetes added the experimental horizontal autoscaling of pods,
ensuring the number of pods automatically adjusts as applications’ demands rise
and fall. This autoscaling, described in detail in the [documentation]
(../user-guide/horizontal-pod-autoscaler.md), works based on the following
calculation:

```
TargetNumOfPods = ceil(sum(CurrentPodsResourceConsumption) / TargetResourceConsumption)
```

If the selected pods are operating outside of their target resource
consumption, the autoscaler will destroy or create pods until all the selected
pods are operating in accordance with target consumption.

Overall, this method of autoscaling works well, yet it could
be improved by accounting for the amount of time necessary
to initialize a pod (i.e. for a pod to be ready to share the computational
work). A pod’s initialization time can be lengthy if the pod must initialize
a framework, download shard files from a
network, register itself with some type of manager, etc.
More specifically, if the newly created pods take a long time to
initialize, the already existing pods will spend the entire
initialization time operating outside of target
resource consumption while they wait for the newly created pods to
share the computational work. Depending on pod initialization time, and the
danger of operating outside target resource consumption, it can be
extremely detrimental to not account for pod initialization time.

**Predictive horizontal pod autoscaling** addresses this problem.
It accounts for pod initialization time by
determining the target number of pods based on the predicted future pod resource
consumption, instead of the current pod resource consumption. In other words,
if it takes 10 minutes for a pod to initialize, than if it is 6:00pm and
predictive autoscaling is enabled, autoscaling behavior is determined by a
prediction of the application’s demands at 6:10pm. If it is determined that a
new instance of this pod is needed based on the prediction, the new pod will be
created at 6:00pm, meaning it will be initialized by 6:10pm. The pods will
never have to operate outside of their target resource consumption (as opposed
to if autoscaling was occurring in a non-predictive manner, in which case the
pod would not be created until 6:10 and would not be initialized until 6:20pm,
meaning there was 10 minutes in which pods were operating outside of target
resource consumption). Predictive horizontal pod autoscaling can
decrease the impact of pod initialization times on autoscaling responsiveness and thus
decrease the amount of time pods spend operating outside of their target
resource consumption.

## Predictive Autoscaling Algorithm

Currently, horizontal pod autoscaling is implemented as a control loop.
Every period, the autoscaler checks all machines specified by the
`Status.PodSelector` of the Scale subresource, and checks their metrics through
the [Compute Resource Metrics Api](compute-resource-metrics-api.md).
It computes the `TargetNumOfPods` based on the previously described algorithm, with the following:

```
TargetNumOfPods = ceil(sum(CurrentPodsResourceConsumption) / TargetResourceConsumption)
```

Predictive horizontal autoscaling will use a similar algorithm with one
significant difference. Rather than calculating based on the
`CurrentPodsResourceConsumption`, we will calculate based on
`FuturePodsResourceConsumption`, which we initially define as
`FuturePodsResourceConsumption = CurrentPodResourceConsumption +
(PodsAverageInitializationTime * DerivativeCurrentPodsResourceConsumptionPerSecond)`.
The calculation for `FuturePodsResourceConsumption` could become more complex,
incorporating increasingly complex methods of prediction, in the future.
We now calculate `TargetNumOfPods` with the following:

```
TargetNumOfPods = ceil(sum(FuturePodsResourceConsumption)) / Target
```

The modifications for this option will predominantly occur in
`pkg/controllers/podautoscaler/horizontal.go` in the
`computeReplicasForCPUUtilization` function. The new algorithm will be used
only if predictive auto-scaling is turned on (as discussed in the next
section).

## Implementation

### Turning on Prediction

Initially, the configuration value for enabling predictive autoscaling will be recorded in
the `HorizontalPodAutoscaler.ObjectMeta.Annotations`.

This annotation can be done via the following command:

```sh
kubectl annotate hpa foo predictive=’true’
```

### Other modifications

Additionally, the algorithm for predictive autoscaling requires two new pieces
of information: a pod’s initialization time and the rate of change for the
metric upon which scaling behavior is determined (currently only CPU). These
two values are currently untracked; the following two sections describe the
changes to record them.

#### Recording `AveragePodInitializationTime`

`AveragePodInitializeTime` has a couple of important nuances. First, it is not
simply the amount of time between when a pod is ordered to be created and when
it is created. Rather, we want to record the amount of time from when a pod is
ordered to be created and when it is ready to perform the task for which it is
created. For example, a pod may be created, but if the purpose of the pod is to
run a webserver, and that web server takes five minutes to be able to serve
requests, then the mere fact that the pod is created does not help balance the
load nor does it accomplish the purpose of autoscaling. It is the difference
between the pod’s creation time and the time at which the pod is ready to
perform its task that we are interested in; we call this metric
`InitializationTime`.

We will calculate `AveragePodInitializationTime` by storing a measure of
`InitializationTime` for each pod created during auto-scaling in the pod’s
`Annotations`. We can then use the pod selector specified by the horizontal pod
auto-scaler to retrieve and average these values, resulting in the
`AverageInitializationTime` metric.

We ensure each newly created pod tracks `InitializationTime` in
`ObjectMeta.Annotations` by adding (in
`pkg/controller/podautoscaler/horizontal.go`) a watcher for any modifications
to pods that are created by the horizontal autoscaler. If the modification
event for these pods indicates that they are now “ready”, then we set the
`InitializationTime` to be the difference between the time at which they became
ready and the time at which the pod was ordered to be created.

It would then be possible to calculate the `AveragePodInitializationTime` for pods
in a horizontal controller group by listing all pods created by this autoscaler
and averaging their `InitializationTime`.

#### Recording `CPUUtilizationDerivative`

Our proposed algorithm for predictive scaling requires us to know the
derivative of the metric we are using to determine scaling behavior, in this
case `CPUUtilizationDerivative`. Thus, we will need to modify the
`HorizontalPodAutoscalerStatus` struct in `pkg/apis/extensions/v1beta1/types.go` to record one extra field that will support the performance of a simple derivative calculation.

```go
type HorizontalPodAutoscalerStatus struct {
...

    // current average CPU utilization over all pods...
    CurrentCPUUtilizationPercentage *int `json:”currentCPUUtilizationPercentage ...”`

    // previous average CPU utilization over all pods…
    PreviousCPUUtilizationPercentage *int `json:”previousCPUUtilizationPercentage…”`

}
```

Recording this field would require the following general modifications to
record the `CurrentCPUUtilizationPercentage` in the
`PreviousCPUUtilizationPercentage` field before updating
`CurrentCPUUtilizationPercentage`.

With the previous cpu utilization percentage and the time interval at which
measurements are taken, it is possible to perform a very simple derivative
calculation necessary for the predictive scaling algorithm.

This is the simplest method of determining the derivative; naturally, a more
complex method could be implemented. For example, instead of tracking only a
single `PreviousCPUUtilizationPercentage`, we could track a slice of size n of
previous utilizations. We would then calculate the derivative using the n
observations, as opposed to just 2. In the spirit of only adding what is
strictly necessary, this more complex method will only be added if strictly
necessary.

## Configuration

To make it easy to add prediction to a `HorizontalPodAutoscaler`, we simply
make it a configurable option. See the *Turning on Prediction* section for further
information on implementation. As this feature stabilizes/if it becomes
particularly popular, it could become part of the options specified through
`kubectl autoscale` or even become the default if it is always advantageous.

## Implementation Plan

This proposal can be implemented in a number of steps, such that there are
multiple, non-breaking PR's before this feature is added.

- [  ] Add the `PreviousCPUUtilizationPercentage` metric as described above.
- [  ] Add the tracking of `AverageInitializationTime` to the
 `podautoscaler/horizontal.go` controller.
- [  ] Modify `podautoscaler/horizontal.go` to auto-scale predictively if the
 option is turned on.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/predictive-horizontal-pod-autoscaling.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
