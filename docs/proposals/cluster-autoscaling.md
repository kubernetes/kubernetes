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
[here](http://releases.k8s.io/release-1.0/docs/proposals/cluster-autoscaling.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Cluster Autoscaling

## Preface

This document briefly describes the design and roadmap of cluster autoscaling in Kubernetes.

## Overview

The purpose of Cluster Autoscaler is to adjust the size of the Kubernetes cluster to match
the current needs of the applications running inside of the cluster. In general, if cluster
nodes are under heavy load or there are pods that cannot be scheduled due to insufficient
free resources then more machines should be added to the cluster. In a similar way if the
utilization on the machines is low and it is apparent that the same set of pods could be run on
fewer nodes, then the size of the cluster should be reduced.

## Kubernetes 1.1 - no Kubernetes-native cluster autoscaling

In the current Kubernetes version we have a basic support for cluster autoscaling. It is based
on Google Cloud Autoscaler (https://cloud.google.com/compute/docs/autoscaler/) and works only
when Kubernetes is running on Google Compute Engine. It looks at total cpu usage on each node (including system stuff)
and total cpu request of all running pods. It can also scale on memory but for the simplicity
this document will focus only on the CPU usage aspect (other metrics work in the exactly same way).
The user is expected to provide the target level of utilization (which is now common for all metrics)
and GCA will adjust the cluster size to meet the target. From the bird’s eye the architecture looks as below.

```
                  +--MIG-----------------------------+
                  |                                  |
                  | +--------+ +--------+ +--------+ |
                  | |        | |        | |        | |
                  | | Node1  | | Node2  | | NodeX  | |
                  | |        | |        | |        | +<--+
                  | +--------+ +--------+ +--------+ |   |
                  | |cAdvisor| |cAdvisor| |cAdvisor| |   |
                  | +--------+ +--------+ +----+---+ |   |
                  |     |          |           |     |   |
                  +-----|----------|-----------|-----+   |
                        |          |           |         |
                        |      +---v----+      |         |
                        +----> |Heapster| <----+         |
                               +---+----+                |
                                   |                     |
   Kubernetes                      |                     |
-----------------------------------|---------------------|------
   Google                          |                     |
                                   |                     |
                            +------v-----+       +-------+------+
                            |Google Cloud|       |MIG Definition|
                            | Monitoring |       +-------^------+
                            +------------+               |
                                   |                     |
                                   |                     |
                            +------v-----+               |
                +------+    |Google Cloud+---------------+
                |Config|--->| Autoscaler |
                +------+    +------------+
```

CAdvisor instances gather metrics for nodes they are running on. Periodically Heapster (a service running
on one of the nodes) queries all of the available nodes/cAdvisors, aggregates the metrics and pushes the
usage to Google Cloud Monitoring (Heapster also takes Pod requests from apiserver but this fact is not
depicted on the diagram). Google Cloud Autoscaler checks the metrics in GCM, compares the values
and min/max cluster sizes from the config (that is passed through google-specific gcloud tool or web ui,
and currently set up in kube-up/push) and updates the number of nodes in MIG (Managed Instance Group)
definition. MIG then starts (or stops) the nodes.


The current solution has a bunch of issues:

* (I1) [GCE-specific API] Autoscaler configuration is Google-specific thus on other cloud providers the configuration may look completely different.
* (I2) [GCE-specific logic] Autoscaler logic (scale on a custom metric) is Google-specific - it might be hard to get it working on AWS or elsewhere
because their autoscaler either does not allow to use custom metric or interprets/acts differently.
* (I3) [Works well only for well-formed pods]
Autoscaler grows the cluster only in reaction to resource consumption of pods that are already running, i.e.
it does not take into account pods that are pending. It still tends to increase the probability that
a pod will schedule, by keeping the cluster "not too full" (reasonable amount of free resources), but
it does not guarantee to prevent pods that stay pending indefinitely - for example, the resources might be
too fragmented to meet the pod's requirements, or the pod might have a node selector that can't be satisfied
by any of the nodes that have sufficient free resources. More specifically on the first point, cluster autoscaler guarantees
scheduling only the pods that are smaller than `1-target` cpu-request-utilization. If the target is `75% (* machine-cpu-count)`
 then only pods of size `25% * machine-cpu-count` will schedule (assuming that they don’t have other serious constraints).
Bigger pods scheduling is best-effort - autoscaler may keep it unscheduled forever. (see small tasks in appendix at the end of this document)
* (I4) [Autoscaler kills nodes at random when scaling down] K8S does not influence which node is killed when scaling down.
* (I5) [Hard to debug] Autoscaler decisions are not propagated back to Kubernetes as events.
* (I6) [Hard to enable/disable] Autoscaler setup takes place on cluster startup.
* (I7) [Heapster won't schedule] All scaling decisions require Heapster to be up and running. In case of a very
unfortunate scale down (I4) combined with new tasks it may be possible that Heapster will not
start again leaving the cluster in the fixed, too small size.

## Kuberntes 1.2 - small KAC

For Kubernetes 1.2 we plan to modify the architecture slightly. The main difference is moving the config to the Kubernetes world.
The config will be placed in apiserver (or in the other config storage that will be ready for 1.2)  and will be handled by
kubernetes-specific tools (kubectl or so). A new kubernetes controller will be added - Kubernetes Autoscaler for Cluster (abbreviated as KAC).

In version 1.2 we will provide “small” KAC. Its responsibility will be mainly limited to reading the config and updating Google
Cloud Autoscaler accordingly. It will also observe changes in MIG and generate appropriate kubernetes events.

```
                          +--MIG-----------------------------+
                          |                                  |
                          | +--------+ +--------+ +--------+ |
                          | |        | |        | |        | |
                          | | Node1  | | Node2  | | NodeX  | |
                          | |        | |        | |        | +<--+
                          | +--------+ +--------+ +--------+ |   |
            +---------+   | |cAdvisor| |cAdvisor| |cAdvisor| |   |
            |  Config |   | +--------+ +--------+ +----+---+ |   |
            +---------+   |     |          |           |     |   |
                 |        +-----|----------|-----------|-----+   |
                 |              |          |           |         |
            +----v----+         |      +---v----+      |         |
            |   KAC   |         +----> |Heapster| <----+         |
            +---------+                +---+----+                |
                 |                         |                     |
 Kubernetes      |                         |                     |
-----------------|-------------------------|---------------------|------
  Google         |                         |                     |
                 |                         |                     |
                 |                  +------v-----+       +-------+------+
                 |                  |Google Cloud|       |MIG Definition|
                 |                  | Monitoring |       +-------^------+
                 |                  +------------+               |
                 |                         |                     |
                 |                         |                     |
                 |                  +------v-----+               |
                 +----------------->|Google Cloud+---------------+
                                    | Autoscaler |
                                    +------------+
```


We will also configure GCE Autoscaler so that it deletes nodes  in a non-random way. There are couple strategies for killing to be considered:

* Node with the fewest number of pods.
* Node with the fewest number of non-replicated pods.
* Node with the lowest load.
* Node with fewest number of guaranteed-QoS pods.

GCA will know about our preferences from the metrics that would go to GCM the similar way as cpu usage (it will kill nodes starting from
the one with the lowest metric). As for 1.2 the corresponding API (for selecting which metric to use to choose the victim)
may not be ready we will use a predefined metric name and suffix for GCA name.
We will use MIG shutdown script to support pod graceful termination in case of node deletion.

This will solve the following problems:

* (I1) [GCE specific API] we will have a single tool for all autoscalers (kubectl) with defined configuration
* (I4) [Autoscaler kills nodes at random] We will use simplest metric (number of pods per node); based on user
feedback we may enhance it and include more advanced logic by computing per node metrics inside KAC (to take
into account what pods are running on each node). It will also help greatly with (I7).
* (I5) [Hard to debug] KAC will generate events that correspond to GCA decisions based on gcloud compute operations list.
* (I6) [Hard to enable/disable] user can easily update the configuration on a running cluster.

It will also allow us to switch the implementation later without changing the interface (too much).

This will not solve the following problems:

(I2) [GCE specific logic] To port this solution to other cloud providers one would have to reimplement whole logic that
implements common interface (or configure cloud-provider specific autoscaler); it’s possible but hard.
(I3) [Works only for well formed pods] We will also introduce an experimental/hacky metrics for pending pods that will be useful
for scaling up clusters when the pods are larger than 1-target. This will be only a help, not a full solution.

For development/experimentation/proof-of-concept KAC config it will be stored in Etcd as an API object. We will consider
other storage options once they become available.

## Kubernetes 1.3 - medium KAC

With medium KAC the goal is to replace all of the Google-specific parts of the autoscaling infrastructure
with open-source code that provides the same functionality. This will make cluster autoscaling cloud-provider
independent (though an adapter layer will be needed for each cloud provider). It will take the required metrics from Heapster.


```
                          +--Nodes---------------------------+
                          |                                  |
                          | +--------+ +--------+ +--------+ |
                          | |        | |        | |        | |
                          | | Node1  | | Node2  | | NodeX  | |
                          | |        | |        | |        | +<--+
                          | +--------+ +--------+ +--------+ |   |
            +---------+   | |cAdvisor| |cAdvisor| |cAdvisor| |   |
            |  Config |   | +--------+ +--------+ +----+---+ |   |
            +---------+   |     |          |           |     |   |
                 |        +-----|----------|-----------|-----+   |
                 |              |          |           |         |
            +----v----+         |      +---v----+      |         |
            |   KAC   |<-+      +----> |Heapster| <----+         |
            +---------+  |             +---+----+                |
                 |       |                 |                     |
                 |       +-----------------+                     |
 Kubernetes      |                                               |
-----------------|-----------------------------------------------|------
 Cloud provider  |                                               |
                 |                                               |
                 |                                       +-------+-------+
                 +-------------------------------------> |Node controller|
                                                         +---------------+
```

This will solve the following problems:
(I2) [GCE specific logic] As cluster autoscaling will have most of its components inside the cluster it will be provider independent (I2).

At this point we will encourage the community to contribute code that talks to particular cloud provider node controller.

## Kubernetes 1.4-1.5 - big KAC

Assuming the need for more sophisticated cluster autoscaling we will provide a full blown autoscaler that will solve (I3)[Works only for well formed pods] and will:

* Take into account the number of pending pods and will check whether adding a node will allow to schedule them.
* Choose the best node to kill once the downscale decision was made (and before will make sure that the homeless pods will schedule somewhere).
* Reasonably handle batch-like best-effort workload
* Use rescheduler to redistribute the load (https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/rescheduler.md).

All of that will happen inside of the Kubernetes cluster. The exact details are to be decided.


## Appendix

## Small task scheduling guarantee

Each pod may provide CPU request information - minimal amount of cpu that is required for pod to run.
It may consume more, if available but it will have this amount guaranteed. On a machine with 1 CPU we can
put pods that request for 0.2, 0.3, 0.4 CPU respectively. The total amount of requested CPU will be then 0.9.

In autoscaler we can specify that nodes should have pods with total amount of X CPU, on average.
If this average is higher, a new node is created and new pods will most likely go there. Average of
X CPU guarantees that there there is a node whose CPU is committed to less or equal to X. Thus there is
always a node that can fit a pod with request set to `1-X`.

If a pod has some bigger request then there is no guarantee. It may be a problem for customers who would
like to have big pods running on nodes or even devote nodes to one type of job (some kind of database for instance).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/cluster-autoscaling.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
