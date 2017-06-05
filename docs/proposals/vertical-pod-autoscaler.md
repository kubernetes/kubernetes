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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/vertical-pod-autoscaler.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

Vertical Pod Autoscaler is a feature that periodically adjust pods' [Resources](resource-qos.md#resource-specifications) based on demand signals.
This is a natural extension of [Initial Resources](initial-resources.md) plugin.

**Note:** since there is no plan to implement the feature immediately (as for 9/9/15) this document is just an overview how it should be done
rather than detailed design.

## Motivation

Setting Resources field incorrectly could lead to wasted resources, violating service latency objectives,
or one or more containers being killed because the machine ran out of memory. Initial Resources plugin
set Resources but it can be wrong, especially in case when there is not enough data for the proper prediction, so
there is need to add possibility to adjust the decision later (possibly many times in the fullness of time)
using complex data mining of historical behavior. The intention is to make the adjustments relatively infrequently.

## Overview

Vertical Pod Autoscaler will consist of two parts: the Prediction actutation logic which decides when the prediction should be made and actuates the decision
and Prediction API which will provide resource predictions for pods.

### Prediction actuation

It will consist of two parts:

* Initial Resources admission plugin, described in [a different proposal](initial-resources.md)
* Vertical Pod Autoscaler controller, which periodically adjusts Resources (set by user or Initial Resources)

Both components will query Prediction API (see details below) for resource predictions.

### Prediction API

Prediction API will be an API plugin in API Server backed by an add-on (initially by Heapster). The API will return resource prediction
based on PodSpec (and possibly other params). The predicting engine will be easily replaceable by changing
the setting of API Server to point to another add-on. Prediction API may be also used by other features like building an UI.

### Missing features in Kubernetes

Kubernetes doesn't yet support possibility to actuate Vertical Pod Autoscaler's decisions in-place (without rescheduling).
There is no possibility to update Resources in PodSpec and no support to change resource limits on Docker.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/vertical-pod-autoscaler.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
