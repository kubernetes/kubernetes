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

# Predictive Horizontal Pod Autoscaling

**Author**: Matt McNaughton (@mattjmcnaughton)

## Preface

This document describes the addition of prediction to the
[horizontal pod autoscaler] (../user-guide/horizontal-pod-autoscaler.md).
This addition will work to improve Kubernetes' auto-scaling efficient resource
utilization (ERU) and quality of service (QOS). This work was undertaken as part
of my undergraduate honor's
[thesis](https://github.com/mattjmcnaughton/thesis/blob/master/writing/thesis.pdf),
and seeks to become part of
[kubernetes/contrib](https://github.com/kubernetes/contrib). The implementation
can be found on [this
fork](https://github.com/mattjmcnaughton/kubernetes/tree/add-predictive-autoscaling),
and all pertinent changes can be found within [this
directory](https://github.com/mattjmcnaughton/kubernetes/blob/add-predictive-autoscaling/pkg/controller/podautoscaler/).

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
work). For the typical web server, this initialization time is short (i.e. no more than five seconds).
However, a pod’s initialization time can be lengthy if the pod must initialize
a framework, download shard files from a
network, register itself with some type of manager, etc. For example,
initializing an ElasticSearch database with a 25MB seed file took approximately
135s. If the newly created pods take a long time to
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

It computes the `TargetNumOfPods` based on the previously described algorithm, with the following:

```
TargetNumOfPods = ceil(sum(CurrentPodsResourceConsumption) / TargetResourceConsumption)
```

Predictive horizontal autoscaling will use a similar concept with one
significant difference. Rather than calculating based on the
`CurrentPodsResourceConsumption`, we will calculate based on
`FuturePodsResourceConsumption`.

We define `FuturePodsResourceConsumption` as the resource consumption at time `t`, where

```
t = CurrentTime + PodInitializationTime
```.

We additionally let the function `LineOfBestFit` be a line of best fit for a
plotting of observation time, `x`, and Previous CPU Utilizations, `y`.

We then use `t` and `LineOfBestFit` to define:

```
FutureCPUUtilization = LineOfBestFit(t)
```.

The code implementing this algorithm can be found in
[horizontal.go](https://github.com/mattjmcnaughton/kubernetes/blob/add-predictive-autoscaling/pkg/controller/podautoscaler/horizontal.go),
specifically this `predictCPUUtilization`
[method](https://github.com/mattjmcnaughton/kubernetes/blob/add-predictive-autoscaling/pkg/controller/podautoscaler/horizontal.go#L523).

## Implementation

In order to enable Horizontal Pod Auto-scaling, we must add methods for
calculating both `AveragePodInitializationTime` and the `LineOfBestFit` for
`PreviousCPUUtilization`, as well as a method for turning predictive
auto-scaling on and off.

### Turning on Prediction

Initially, the configuration value for enabling predictive autoscaling will be recorded in
the `HorizontalPodAutoscaler.ObjectMeta.Annotations`.

This annotation can be done via the following command:

```sh
kubectl annotate hpa foo predictive=’true’
```

The implementation can be seen in the `isPredictive`
[function](https://github.com/mattjmcnaughton/kubernetes/blob/add-predictive-autoscaling/pkg/controller/podautoscaler/predictive_utils.go#L48).

### Recording `AveragePodInitializationTime`

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
`InitializationTime`. We consider a pod to be initialized if its [readiness
probe](http://kubernetes.io/docs/user-guide/production-pods/#liveness-and-readiness-probes-aka-health-checks)
indicates readiness.

We calculate the `AveragePodInitializationTime` as follows:

For all pods controller by the auto-scaler, we select only those that are
currently in the `ready` state. If an initialization time has not been recorded
for said pod, we subtract the `pod.creationTimestamp` from the time at which the
pod switched into the `Ready` state. We then add all of the `InitializationTimes` together,
and divide by the number of pods, in order to get the
`AveragePodInitializationTime`.

The implementation can be seen in the `averagePodInitializationTime`
[method](https://github.com/mattjmcnaughton/kubernetes/blob/add-predictive-autoscaling/pkg/controller/podautoscaler/horizontal.go#L483).

### Determining `LineOfBestFit` for `PreviousCPUUtilizations`

First, in order to create the `LineOfBestFit` for `PreviousCPUUtilizations`, we
must first find a method of storing `PreviousCPUUtilizations`. Each time we
use CPU utilization to make an auto-scaling decision, we record it to `hpa.Annotations` with a
timestamp, as can be seen in the `recordCPUUtilization`
[method](https://github.com/mattjmcnaughton/kubernetes/blob/add-predictive-autoscaling/pkg/controller/podautoscaler/horizontal.go#L443). The number of observations is limited by a constant
multiple of `PodInitilizationTime`.

We then calculate a linear `LineOfBestFit` using these previous observations, as
can be seen in the `predictCPUUtilization`
[method](https://github.com/mattjmcnaughton/kubernetes/blob/add-predictive-autoscaling/pkg/controller/podautoscaler/horizontal.go#L523). There are two important assumptions in place. First, we only currently
auto-scale if the `predictedCPUUtilization` is greater than the
`currentCPUUtilization`, because we are not concerned about
`PodInitilizationTime` having adverse side effects when downscaling. Second, we
assume that resource utilization varies linearly. This assumption may be
inaccurate for certain traffic patterns, but fortunately we have implemented
predictive auto-scaling such that it is easy to plug-in multiple different
modelling methods.

## Evaluation

Overall, predictive auto-scaling is advantageous in some situations, and
disadvantageous in others, depending on the traffic pattern and the method of
modelling the `LineOfBestFit`. Chapter 5 of my
[thesis](https://github.com/mattjmcnaughton/thesis/blob/master/writing/thesis.pdf) provides
a detailed evaluation for anyone interested.

## Thank you!

Thank you to Brendan Burns for guiding me in implementing this feature and for
serving as my second reader, and thank you to the community for all your
assistance throughout the process!

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/predictive-horizontal-pod-autoscaling.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
