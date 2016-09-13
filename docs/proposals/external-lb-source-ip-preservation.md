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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.4/docs/proposals/external-lb-source-ip-preservation.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Overview](#overview)
  - [Motivation](#motivation)
- [Alpha Design](#alpha-design)
  - [Overview](#overview-1)
  - [Traffic Steering using LB programming](#traffic-steering-using-lb-programming)
  - [Traffic Steering using Health Checks](#traffic-steering-using-health-checks)
  - [Choice of traffic steering approaches by individual Cloud Provider implementations](#choice-of-traffic-steering-approaches-by-individual-cloud-provider-implementations)
  - [API Changes](#api-changes)
    - [Local Endpoint Recognition Support](#local-endpoint-recognition-support)
    - [Service Annotation to opt-in for new behaviour](#service-annotation-to-opt-in-for-new-behaviour)
    - [NodePort allocation for HealthChecks](#nodeport-allocation-for-healthchecks)
  - [Behavior Changes expected](#behavior-changes-expected)
    - [External Traffic Blackholed on nodes with no local endpoints](#external-traffic-blackholed-on-nodes-with-no-local-endpoints)
    - [Traffic Balancing Changes](#traffic-balancing-changes)
  - [Cloud Provider support](#cloud-provider-support)
    - [GCE 1.4](#gce-14)
      - [GCE Expected Packet Source/Destination IP (Datapath)](#gce-expected-packet-sourcedestination-ip-datapath)
      - [GCE Expected Packet Destination IP (HealthCheck path)](#gce-expected-packet-destination-ip-healthcheck-path)
    - [AWS TBD](#aws-tbd)
    - [Openstack TBD](#openstack-tbd)
    - [Azure TBD](#azure-tbd)
  - [Testing](#testing)
- [Beta Design](#beta-design)
  - [API Changes from Alpha to Beta](#api-changes-from-alpha-to-beta)
- [Future work](#future-work)
- [Appendix](#appendix)

<!-- END MUNGE: GENERATED_TOC -->

# Overview

Kubernetes provides an external loadbalancer service type which creates a virtual external ip
(in supported cloud provider environments) that can be used to load-balance traffic to
the pods matching the service pod-selector.

## Motivation

The current implemention requires that the cloud loadbalancer balances traffic across all
Kubernetes worker nodes, and this traffic is then equally distributed to all the backend
pods for that service.
Due to the DNAT required to redirect the traffic to its ultimate destination, the return
path for each session MUST traverse the same node again. To ensure this, the node also
performs a SNAT, replacing the source ip with its own.

This causes the service endpoint to see the session as originating from a cluster local ip address.
*The original external source IP is lost*

This is not a satisfactory solution - the original external source IP MUST be preserved for a
lot of applications and customer use-cases.

# Alpha Design

This section describes the proposed design for
[alpha-level](../../docs/devel/api_changes.md#alpha-beta-and-stable-versions) support, although
additional features are described in [future work](#future-work).

## Overview

The double hop must be prevented by programming the external load balancer to direct traffic
only to nodes that have local pods for the service. This can be accomplished in two ways, either
by API calls to add/delete nodes from the LB node pool or by adding health checking to the LB and
failing/passing health checks depending on the presence of local pods.

## Traffic Steering using LB programming

This approach requires that the Cloud LB be reprogrammed to be in sync with endpoint presence.
Whenever the first service endpoint is scheduled onto a node, the node is added to the LB pool.
Whenever the last service endpoint is unhealthy on a node, the node needs to be removed from the LB pool.

This is a slow operation, on the order of 30-60 seconds, and involves the Cloud Provider API path.
If the API endpoint is temporarily unavailable, the datapath will be misprogrammed till the
reprogramming is successful and the API->datapath tables are updated by the cloud provider backend.

## Traffic Steering using Health Checks

This approach requires that all worker nodes in the cluster be programmed into the LB target pool.
To steer traffic only onto nodes that have endpoints for the service, we program the LB to perform
node healthchecks. The kube-proxy daemons running on each node will be responsible for responding
to these healthcheck requests (URL `/healthz`) from the cloud provider LB healthchecker. An additional nodePort
will be allocated for these health check for this purpose.
kube-proxy already watches for Service and Endpoint changes, it will maintain an in-memory lookup
table indicating the number of local endpoints for each service.
For a value of zero local endpoints, it responds with a health check failure (503 Service Unavailable),
and success (200 OK) for non-zero values.

Healthchecks are programmable with a min period of 1 second on most cloud provider LBs, and min
failures to trigger node health state change can be configurable from 2 through 5.

This will allow much faster transition times on the order of 1-5 seconds, and involve no
API calls to the cloud provider (and hence reduce the impact of API unreliability), keeping the
time window where traffic might get directed to nodes with no local endpoints to a minimum.

## Choice of traffic steering approaches by individual Cloud Provider implementations

The cloud provider package may choose either of these approaches. kube-proxy will provide these
healthcheck responder capabilities, regardless of the cloud provider configured on a cluster.

## API Changes

### Local Endpoint Recognition Support

To allow kube-proxy to recognize if an endpoint is local requires that the EndpointAddress struct
should also contain the NodeName it resides on. This new string field will be read-only and
populated *only* by the Endpoints Controller.

### Service Annotation to opt-in for new behaviour

A new annotation `service.alpha.kubernetes.io/external-traffic` will be recognized
by the service controller only for services of Type LoadBalancer. Services that wish to opt-in to
the new LoadBalancer behaviour must annotate the Service to request the new ESIPP behavior.
Supported values for this annotation are OnlyLocal and Global.
- OnlyLocal activates the new logic (described in this proposal) and balances locally within a node.
- Global activates the old logic of balancing traffic across the entire cluster.

### NodePort allocation for HealthChecks

An additional nodePort allocation will be necessary for services that are of type LoadBalancer and
have the new annotation specified. This additional nodePort is necessary for kube-proxy to listen for
healthcheck requests on all nodes.
This NodePort will be added as an annotation (`service.alpha.kubernetes.io/healthcheck-nodeport`) to
the Service after allocation (in the alpha release). The value of this annotation may also be
specified during the Create call and the allocator will reserve that specific nodePort.


## Behavior Changes expected

### External Traffic Blackholed on nodes with no local endpoints

When the last endpoint on the node has gone away and the LB has not marked the node as unhealthy,
worst-case window size = (N+1) * HCP, where N = minimum failed healthchecks and HCP = Health Check Period,
external traffic will still be steered to the node. This traffic will be blackholed and not forwarded
to other endpoints elsewhere in the cluster.

Internal pod to pod traffic should behave as before, with equal probability across all pods.

### Traffic Balancing Changes

GCE/AWS load balancers do not provide weights for their target pools. This was not an issue with the old LB
kube-proxy rules which would correctly balance across all endpoints.

With the new functionality, the external traffic will not be equally load balanced across pods, but rather
equally balanced at the node level (because GCE/AWS and other external LB implementations do not have the ability
for specifying the weight per node, they balance equally across all target nodes, disregarding the number of
pods on each node).

We can, however, state that for NumServicePods << NumNodes or NumServicePods >> NumNodes, a fairly close-to-equal
distribution will be seen, even without weights.

Once the external load balancers provide weights, this functionality can be added to the LB programming path.
*Future Work: No support for weights is provided for the 1.4 release, but may be added at a future date*

## Cloud Provider support

This feature is added as an opt-in annotation.
Default behaviour of LoadBalancer type services will be unchanged for all Cloud providers.
The annotation will be ignored by existing cloud provider libraries until they add support.

### GCE 1.4

For the 1.4 release, this feature will be implemented for the GCE cloud provider.

#### GCE Expected Packet Source/Destination IP (Datapath)

- Node: On the node, we expect to see the real source IP of the client. Destination IP will be the Service Virtual External IP.

- Pod: For processes running inside the Pod network namepsace, the source IP will be the real client source IP. The destination address will the be Pod IP.

#### GCE Expected Packet Destination IP (HealthCheck path)

kube-proxy listens on the health check node port for TCP health checks on :::.
This allow responding to health checks when the destination IP is either the VM IP or the Service Virtual External IP.
In practice, tcpdump traces on GCE show source IP is 169.254.169.254 and destination address is the Service Virtual External IP.

### AWS TBD

TBD *discuss timelines and feasibility with Kubernetes sig-aws team members*

### Openstack TBD

This functionality may not be introduced in Openstack in the near term.

*Note from Openstack team member @anguslees*
Underlying vendor devices might be able to do this, but we only expose full-NAT/proxy loadbalancing through the OpenStack API (LBaaS v1/v2 and Octavia). So I'm afraid this will be unsupported on OpenStack, afaics.

### Azure TBD

*To be confirmed* For the 1.4 release, this feature will be implemented for the Azure cloud provider.

## Testing

The cases we should test are:

1. Core Functionality Tests

1.1 Source IP Preservation

Test the main intent of this change, source ip preservation - use the all-in-one network tests container
with new functionality that responds with the client IP. Verify the container is seeing the external IP
of the test client.

1.2 Health Check responses

Testcases use pods explicitly pinned to nodes and delete/add to nodes randomly. Validate that healthchecks succeed
and fail on the expected nodes as endpoints move around. Gather LB response times (time from pod declares ready to
time for Cloud LB to declare node healthy and vice versa) to endpoint changes.

2. Inter-Operability Tests

Validate that internal cluster communications are still possible from nodes without local endpoints. This change
is only for externally sourced traffic.

3. Backward Compatibility Tests

Validate that old and new functionality can simultaneously exist in a single cluster. Create services with and without
the annotation, and validate datapath correctness.

# Beta Design

The only part of the design that changes for beta is the API, which is upgraded from
annotation-based to first class fields.

## API Changes from Alpha to Beta

Annotation `service.alpha.kubernetes.io/node-local-loadbalancer` will switch to a Service object field.

# Future work

Post-1.4 feature ideas. These are not fully-fleshed designs.



# Appendix

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/external-lb-source-ip-preservation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
