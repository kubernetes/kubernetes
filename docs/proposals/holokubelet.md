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
[here](http://releases.k8s.io/release-1.0/docs/proposals/rescheduler.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# HoloKubelet proposal

## Goal of this document
The goal of this document is to discuss the detailed design of HoloKubelet - a library that can mimic Kubelets behavior without actually doing anything complex (read: CPU or memory consuming).

## Objective of HoloKubelet
HoloKubelet is supposed to be used in Kubemark system as a light-weight replacement for the Kubelet. It will be able to fool Master components into thinking it’s a real
Kubelet, so they’ll treat it as real one. It has to be as light as possible to allow running as many replicas on a single machine as possible, but in the same time needs to
generate as “realistic” load as possible. There’s obvious trade-off between efficiency and accuracy. For now we concentrate more on efficiency and aim at being able to simulate
1000 Node cluster on less than 10 ‘real’ machines equivalent to GCP n1-standard-1. The most important design assumption is that HoloKubelet is supposed to be used for
performance not correctness testing, i.e. it won’t need to verify that the data it gets from the API server make any sense.

## Kubelet analysis
Before designing HoloKubelet itself we need to understand how real Kubelet interacts with API server and other Master components through it.

Currently Kubelet creates 3 Watch clients: for Pods, for Nodes and for Services. Starting from the end:
- Kubelet watches Services only to create environment variables for MY_SERVICE_HOST etc. They are for use of Pods running under this Kubelet, so are of no use to us. Yey:),
- Kubelet watches Nodes, or rather its Node, to get its Nodes definition. This is an important for starting Pods, as in Kubernetes Kubelet is the last arbiter on what is allowed to run on its machine and need to check e.g. NodeLabels. 
- Watching Pods is critical for the Kubelet. Observed Pod operations is the only external influencer of what’s running on Kubelets Node. Kubelet starts/updates/stops Pods according to those changes. As of now Kubelet does not take any actions on its own, except possibly rejecting Pod bindings.

On the active side Kubelet pushes Status updated to the API server:
- for Nodes is once every 10 seconds. This update is just a heartbeat and it just update a timestamp in NodeStatus. This probably will be factored out to separate Heartbeat (or something) object.
- for Pods it’s simpler - Kubelet updates PodStatuses only after it observed some changes in it. Hence as long as noone updated/changes Pod, Kubelet won’t report back its status.

In addition Kubelet will publish Events in case of something happening on the machine:
- multiple events during machine startup
- failed to mount volumes for pod
- kubelet rejects pod for some reason (host port conflict, not enough resources, etc.)
- Kubelet is killing pod because ActiveDeadlineSeconds passed
- failed to pull image
- probe failure
- docker failures
- container creation/startup/death
- failed pod sync
Note that except Event published during Node startup and ones related to Container life-cycle, Kubelet publishes Events only in case of some problems. Because HoloKubelet will simulate “perfect Kubelet” (one that does not have any problems), we can just forget about those Events - I looked at events on our Jenkins scalability cluster and there was exactly 0 ‘error’ Events.

## HoloKubelet design
Note that because our goal is performance testing, we assume that HoloKubelet will act as a perfect Kubelet that does not have any problems ever. In such case, we only need
HoloKubelet to send updated of NodeStatus every 10 seconds and generate Events for Pod start/stop. We probably even can avoid publishing events from Kubelet startup, as it’s
unlikely to influence long-term performance in measurable amount.

If we exclude failure-related and startup Events HoloKubelet will need to:
- Update timestamp in NodeStatus every 10 seconds
- Emit Pulled/Created/Started Events for each created container + one for POD container per Pod
- Emit one Killing Event for each killed container + one for POD container per Pod.
- Update currently is implemented as kill + start, so it’s a combination of two previous points

Above means that HoloKubelet will need to keep in memory list of currently “running” pods and watch for operations on those Pods. This is enough for initial version.

How Kubelet authentication will work TBD.

## Performance analysis
The biggest Kubelet I saw in the wild was using 100MB of memory, and biggest KubeProxy was using about 6MB. I believe it’s reasonable to assume that Kubelete+KubeProxy (adding
some padding) in real world can use around 130MB. It’s reasonable to assume that dry-run in Kubelet and Kube proxy would need ~1/2 resources. This would mean that on a single
n1-standard-1 with 3.75 GB we could run around 50 Holograms per machine (it’s likely an overestimation). If we want to simulate 1000 Node cluster, this would require 20
machines.

On the other hand Holograms should require very little memory, as they would be nearly stateless. HoloProxy would be completely stateless, so it’d use the memory only for
keeping Watch clients open (this should be negligible, but need to test for concrete amount), while HoloKubelet will keep minimal state with the definition of all Pods and it’s
Node. Because we limit number of Pods to 40 per machine, and we can estimate a size of a single Pod/Node api object by few kB, say 5kB per object. This means that the size of a
basic HoloKubelet would be negligible as well (<1MB)

Statically linked Go binary probably uses around 10MB (I need to verify this). If we put both HoloKubelet and HoloProxy in the one binary to reduce the overhead, they should
fit into, say 15MB of memory. This would mean that it’s ~5x gain from the ‘dry-run’ version of the system and we would be able to simulate 1000 Node cluster on ~4 machines 
(+ Master). This obviously assumes that the number of Watch clients per Node won’t be a limiting factor.

## Future
In the future if we decide on extending Kubemark to allow replaying “real” traffic which would include various failures, we’ll need to keep much more state. E.g. we’ll need to
keep in memory time until failure/end per container/pod. This kind of things are out of scope for now.
