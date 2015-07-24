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
[here](http://releases.k8s.io/release-1.0/docs/admin/multi-cluster.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Considerations for running multiple Kubernetes clusters

You may want to set up multiple Kubernetes clusters, both to
have clusters in different regions to be nearer to your users, and to tolerate failures and/or invasive maintenance.
This document describes some of the issues to consider when making a decision about doing so.

Note that at present,
Kubernetes does not offer a mechanism to aggregate multiple clusters into a single virtual cluster. However,
we [plan to do this in the future](../proposals/federation.md).

## Scope of a single cluster

On IaaS providers such as Google Compute Engine or Amazon Web Services, a VM exists in a
[zone](https://cloud.google.com/compute/docs/zones) or [availability
zone](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html).
We suggest that all the VMs in a Kubernetes cluster should be in the same availability zone, because:
  - compared to having a single global Kubernetes cluster, there are fewer single-points of failure
  - compared to a cluster that spans availability zones, it is easier to reason about the availability properties of a
    single-zone cluster.
  - when the Kubernetes developers are designing the system (e.g. making assumptions about latency, bandwidth, or
    correlated failures) they are assuming all the machines are in a single data center, or otherwise closely connected.

It is okay to have multiple clusters per availability zone, though on balance we think fewer is better.
Reasons to prefer fewer clusters are:
  - improved bin packing of Pods in some cases with more nodes in one cluster (less resource fragmentation)
  - reduced operational overhead (though the advantage is diminished as ops tooling and processes matures)
  - reduced costs for per-cluster fixed resource costs, e.g. apiserver VMs (but small as a percentage
    of overall cluster cost for medium to large clusters).

Reasons to have multiple clusters include:
  - strict security policies requiring isolation of one class of work from another (but, see Partitioning Clusters
    below).
  - test clusters to canary new Kubernetes releases or other cluster software.

## Selecting the right number of clusters

The selection of the number of Kubernetes clusters may be a relatively static choice, only revisited occasionally.
By contrast, the number of nodes in a cluster and the number of pods in a service may be change frequently according to
load and growth.

To pick the number of clusters, first, decide which regions you need to be in to have adequate latency to all your end users, for services that will run
on Kubernetes (if you use a Content Distribution Network, the latency requirements for the CDN-hosted content need not
be considered).  Legal issues might influence this as well. For example, a company with a global customer base might decide to have clusters in US, EU, AP, and SA regions.
Call the number of regions to be in `R`.

Second, decide how many clusters should be able to be unavailable at the same time, while still being available.  Call
the number that can be unavailable `U`.  If you are not sure, then 1 is a fine choice.

If it is allowable for load-balancing to direct traffic to any region in the event of a cluster failure, then
you need `R + U` clusters.  If it is not (e.g you want to ensure low latency for all users in the event of a
cluster failure), then you need to have `R * U` clusters (`U` in each of `R` regions).  In any case, try to put each cluster in a different zone.

Finally, if any of your clusters would need more than the maximum recommended number of nodes for a Kubernetes cluster, then
you may need even more clusters.  Kubernetes v1.0 currently supports clusters up to 100 nodes in size, but we are targeting
1000-node clusters by early 2016.

## Working with multiple clusters

When you have multiple clusters, you would typically create services with the same config in each cluster and put each of those
service instances behind a load balancer (AWS Elastic Load Balancer, GCE Forwarding Rule or HTTP Load Balancer) spanning all of them, so that
failures of a single cluster are not visible to end users.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/multi-cluster.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
