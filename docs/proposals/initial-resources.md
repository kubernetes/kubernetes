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
[here](http://releases.k8s.io/release-1.0/docs/proposals/initial-resources.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

Initial Resources is a data-driven feature that based on historical data tries to estimate resource usage of a container without Resources specified
and set them before the container is run. This document describes design of the component.

## Motivation

Since we want to make Kubernetes as simple as possible for its users we don’t want to require setting
[Resources](resource-qos.md#resource-specifications)
for container by its owner. On the other hand having Resources filled is critical for scheduling decisions.
Current solution to set up Resources to hardcoded value has obvious drawbacks. We need to implement a component
which will set initial Resources to a reasonable value.

## Design

InitialResources component will be implemented as an [admission plugin](../../plugin/pkg/admission/) and invoked right before
[LimitRanger](https://github.com/kubernetes/kubernetes/blob/7c9bbef96ed7f2a192a1318aa312919b861aee00/cluster/gce/config-default.sh#L91).
For every container without Resources specified it will try to predict amount of resources that should be sufficient for it.
So that a pod without specified resources will be treated as
[Burstable](resource-qos.md#qos-classes).

InitialResources will set only [request](resource-qos.md#resource-specifications)
(independently for each resource type: cpu, memory)
field in the first version to avoid killing containers due to OOM (however the container still may be killed if exceeds requested resources).
To make the component work with LimitRanger the estimated value will be capped by min and max possible values if defined.
It will prevent from situation when the pod is rejected due to too low or too high estimation.

The container won’t be marked as managed by this component in any way, however appropriate event will be exported.
The predicting algorithm should have very low latency to not increase significantly e2e pod startup latency
[#3954](https://github.com/kubernetes/kubernetes/pull/3954).

### Predicting algorithm details

In the first version estimation will be made based on historical data for the Docker image being run in the container (both the name and the tag matters).
CPU/memory usage of each container is exported periodically (by default with 1 minute resolution) to the backend (see more in [Monitoring pipeline](#monitoring-pipeline)).

InitialResources will set Request for both cpu/mem as the 90th percentile of the first (in the following order) set of samples defined in the following way:

* 7 days same image:tag, assuming there is at least 60 samples (1 hour)
* 30 days same image:tag, assuming there is at least 60 samples (1 hour)
* 30 days same image, assuming there is at least 1 sample

If there is still no data the default value will be set by LimitRanger. Same parameters will be configurable with appropriate flags.

#### Example

If we have at least 60 samples from image:tag over the past 7 days, we will use the 90th percentile of all of the samples of image:tag over the past 7 days.
Otherwise, if we have at least 60 samples from image:tag over the past 30 days, we will use the 90th percentile of all of the samples over of image:tag the past 30 days.
Otherwise, if we have at least 1 sample from image over the past 30 days, we will use that the 90th percentile of all of the samples of image over the past 30 days.
Otherwise we will use default value.

### Monitoring pipeline

In the first version there will be available 2 options for backend for predicting algorithm:

* [InfluxDB](../../docs/user-guide/monitoring.md#influxdb-and-grafana) - aggregation will be made in SQL query
* [GCM](../../docs/user-guide/monitoring.md#google-cloud-monitoring) - since GCM is not as powerful as InfluxDB some aggregation will be made on the client side

Both will be hidden under an abstraction layer, so it would be easy to add another option.
The code will be a part of Initial Resources component to not block development, however in the future it should be a part of Heapster.


## Next steps

The first version will be quite simple so there is a lot of possible improvements. Some of them seem to have high priority
and should be introduced shortly after the first version is done:

* observe OOM and then react to it by increasing estimation
* add possibility to specify if estimation should be made, possibly as ```InitialResourcesPolicy``` with options: *always*, *if-not-set*, *never*
* add other features to the model like *namespace*
* remember predefined values for the most popular images like *mysql*, *nginx*, *redis*, etc.
* dry mode, which allows to ask system for resource recommendation for a container without running it
* add estimation as annotations for those containers that already has resources set
* support for other data sources like [Hawkular](http://www.hawkular.org/)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/initial-resources.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
