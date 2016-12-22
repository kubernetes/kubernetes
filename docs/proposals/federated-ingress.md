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

# Kubernetes Federated Ingress

				  Requirements and High Level Design

							Quinton Hoole

							July 17, 2016

## Overview/Summary

[Kubernetes Ingress](https://github.com/kubernetes/kubernetes.github.io/blob/master/docs/user-guide/ingress.md)
provides an abstraction for sophisticated L7 load balancing through a
single IP address (and DNS name) across multiple pods in a single
Kubernetes cluster. Multiple alternative underlying implementations
are provided, including one based on GCE L7 load balancing and another
using an in-cluster nginx/HAProxy deployment (for non-GCE
environments). An AWS implementation, based on Elastic Load Balancers
and Route53 is under way by the community.

To extend the above to cover multiple clusters, Kubernetes Federated
Ingress aims to provide a similar/identical API abstraction and,
again, multiple implementations to cover various
cloud-provider-specific as well as multi-cloud scenarios. The general
model is to allow the user to instantiate a single Ingress object via
the Federation API, and have it automatically provision all of the
necessary underlying resources (L7 cloud load balancers, in-cluster
proxies etc) to provide L7 load balancing across a service spanning
multiple clusters.

Four options are outlined:

1. GCP only
1. AWS only
1. Cross-cloud via GCP in-cluster proxies (i.e. clients get to AWS and on-prem via GCP).
1. Cross-cloud via AWS in-cluster proxies  (i.e. clients get to GCP and on-prem via AWS).

Option 1 is the:

1. easiest/quickest,
1. most featureful

Recommendations:

+  Suggest tackling option 1 (GCP only) first (target beta in v1.4)
+  Thereafter option 3 (cross-cloud via GCP)
+  We should encourage/facilitate the community to tackle option 2 (AWS-only)

## Options

## Google Cloud Platform only - backed by GCE L7 Load Balancers

This is an option for federations across clusters which all run on Google Cloud Platform (i.e. GCE and/or GKE)

### Features

In summary, all of [GCE L7 Load Balancer](https://cloud.google.com/compute/docs/load-balancing/http/) features:

1. Single global virtual (a.k.a. "anycast") IP address ("VIP" - no dependence on dynamic DNS)
1. Geo-locality for both external and GCP-internal clients
1. Load-based overflow to next-closest geo-locality (i.e. cluster).  Based on either queries per second, or CPU load (unfortunately on the first-hop target VM, not the final destination K8s Service).
1. URL-based request direction (different backend services can fulfill each different URL).
1. HTTPS request termination (at the GCE load balancer, with server SSL certs)

### Implementation

1. Federation user creates (federated) Ingress object (the services
   backing the ingress object must share the same nodePort, as they
   share a single GCP health check).
1. Federated Ingress Controller creates Ingress object in each cluster
   in the federation (after [configuring each cluster ingress
   controller to share the same ingress UID](https://gist.github.com/bprashanth/52648b2a0b6a5b637f843e7efb2abc97)).
1. Each cluster-level Ingress Controller ("GLBC") creates Google L7
   Load Balancer machinery (forwarding rules, target proxy, URL map,
   backend service, health check) which ensures that traffic to the
   Ingress (backed by a Service), is directed to the nodes in the cluster.
1. KubeProxy redirects to one of the backend Pods (currently round-robin, per KubeProxy instance)

An alternative implementation approach involves lifting the current
Federated Ingress Controller functionality up into the Federation
control plane.  This alternative is not considered any any further
detail in this document.

### Outstanding work Items

1. This should in theory all work out of the box.  Need to confirm
with a manual setup. ([#29341](https://github.com/kubernetes/kubernetes/issues/29341))
1. Implement Federated Ingress:
   1. API machinery (~1 day)
   1. Controller (~3 weeks)
1. Add DNS field to Ingress object (currently missing, but needs to be added, independent of federation)
   1. API machinery (~1 day)
   1. KubeDNS support (~ 1 week?)

### Pros

1. Global VIP is awesome - geo-locality, load-based overflow (but see caveats below)
1. Leverages existing K8s Ingress machinery - not too much to add.
1. Leverages existing Federated Service machinery - controller looks
    almost identical, DNS provider also re-used.

### Cons

1. Only works across GCP clusters (but see below for a light at the end of the tunnel, for future versions).

## Amazon Web Services only - backed by Route53

This is an option for AWS-only federations. Parts of this are
apparently work in progress, see e.g.
[AWS Ingress controller](https://github.com/kubernetes/contrib/issues/346)
[[WIP/RFC] Simple ingress -> DNS controller, using AWS
Route53](https://github.com/kubernetes/contrib/pull/841).

### Features

In summary, most of the features of [AWS Elastic Load Balancing](https://aws.amazon.com/elasticloadbalancing/) and [Route53 DNS](https://aws.amazon.com/route53/).

1. Geo-aware DNS direction to closest regional elastic load balancer
1. DNS health checks to route traffic to only healthy elastic load
balancers
1. A variety of possible DNS routing types, including Latency Based Routing, Geo DNS, and Weighted Round Robin
1. Elastic Load Balancing automatically routes traffic across multiple
  instances and multiple Availability Zones within the same region.
1. Health checks ensure that only healthy Amazon EC2 instances receive traffic.

### Implementation

1. Federation user creates (federated) Ingress object
1. Federated Ingress Controller creates Ingress object in each cluster in the federation
1. Each cluster-level AWS Ingress Controller creates/updates
   1. (regional) AWS Elastic Load Balancer machinery which ensures that traffic to the Ingress (backed by a Service), is directed to one of the nodes in one of the clusters in the region.
   1. (global) AWS Route53 DNS machinery which ensures that clients are directed to the closest non-overloaded (regional) elastic load balancer.
1. KubeProxy redirects to one of the backend Pods (currently round-robin, per KubeProxy instance) in the destination K8s cluster.

### Outstanding Work Items

Most of this remains is currently unimplemented ([AWS Ingress controller](https://github.com/kubernetes/contrib/issues/346)
[[WIP/RFC] Simple ingress -> DNS controller, using AWS
Route53](https://github.com/kubernetes/contrib/pull/841).

1. K8s AWS Ingress Controller
1.  Re-uses all of the non-GCE specific Federation machinery discussed above under "GCP-only...".

### Pros

1. Geo-locality (via geo-DNS, not VIP)
1. Load-based overflow
1. Real load balancing (same caveats as for GCP above).
1. L7 SSL connection termination.
1. Seems it can be made to work for hybrid with on-premise (using VPC).  More research required.

### Cons

1. K8s Ingress Controller still needs to be developed. Lots of work.
1. geo-DNS based locality/failover is not as nice as VIP-based (but very useful, nonetheless)
1. Only works on AWS (initial version, at least).

## Cross-cloud via GCP

### Summary

Use GCP Federated Ingress machinery described above, augmented with additional HA-proxy backends in all GCP clusters to proxy to non-GCP clusters (via either Service External IP's, or VPN directly to KubeProxy or Pods).

### Features

As per GCP-only above, except that geo-locality would be to the closest GCP cluster (and possibly onwards to the closest AWS/on-prem cluster).

### Implementation

TBD - see Summary above in the mean time.

### Outstanding Work

Assuming that GCP-only (see above) is complete:

1. Wire-up the HA-proxy load balancers to redirect to non-GCP clusters
1. Probably some more - additional detailed research and design necessary.

### Pros

1. Works for cross-cloud.

### Cons

1. Traffic to non-GCP clusters proxies through GCP clusters.  Additional bandwidth costs (3x?) in those cases.

## Cross-cloud via AWS

In theory the same approach as "Cross-cloud via GCP" above could be used, except that AWS infrastructure would be used to get traffic first to an AWS cluster, and then proxied onwards to non-AWS and/or on-prem clusters.
Detail docs TBD.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/federated-ingress.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
