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

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Node-local services](#node-local-services)
  - [Objective](#objective)
  - [Background](#background)
  - [Detailed discussion](#detailed-discussion)
    - [API changes](#api-changes)
    - [`kube-proxy` changes](#kube-proxy-changes)
      - [iptables kernel mode](#iptables-kernel-mode)
      - [ipvs kernel mode](#ipvs-kernel-mode)
      - [proxy userland mode](#proxy-userland-mode)
    - [n-to-1 forwarding](#n-to-1-forwarding)
  - [Implementation plan](#implementation-plan)
    - [API](#api)
    - [kube-proxy](#kube-proxy)
    - [Documentation](#documentation)
  - [Future work](#future-work)
  - [Security considerations](#security-considerations)

<!-- END MUNGE: GENERATED_TOC -->

# Node-local services

Author: @therc

Date: July 2016

Status: Design in progress

## Objective

Users should be able to create a service that connects pods to a daemon that is
running on the same node (and not elsewhere). Possible use cases:

 - monitoring agents and forwarders such as datadog and
   [tallier](https://github.com/reddit/tallier)
 - connection poolers and service discovery systems such as
   [pgbouncer](https://pgbouncer.github.io/) and
   [synapse](https://github.com/airbnb/synapse)
 - logging agents such as fluentd
 - authenticating proxies such as [aws-es-proxy](https://github.com/kopeio/aws-es-proxy),
   [kube2iam](https://github.com/jtblin/kube2iam) or loasd ([#2209](https://github.com/kubernetes/kubernetes/issues/2209))

At the very least, this new kind of service should support 1:1
_service_->_daemon_ mappings. For cases like `loasd`, it would be ideal to
support n:1 mapping: a number of services all getting forwarded to the same
daemon, with the daemon able to tell which service each connection was meant
for.

## Background

Thanks to DaemonSets, it is possible to schedule daemon pods on every node (or a
specific subset if node labels are specified). Local discovery is more
complicated and currently requires e.g. iptables hacks that bypass Kubernetes
altogether. This is because existing types of services will pick a random
service pod when a client establishes a new connection. More often than not, the
client pod will talk to a service pod running on another machine.

This is not acceptable for some use cases. The daemon might return node-specific
information downstream or report it upstream to a server. If it were to receive
traffic from another machine, data returned or forwarded would be inaccurately
tagged. In other cases, the main concern is reducing cross-node traffic, be it
for bandwidth or security reasons.

Thus, a new service type is needed that always points to the node-local daemon
pod.

## Detailed discussion

Node-local services can reuse most of the existing plumbing.

### API changes

To keep changes to a minimum, a new ServiceAffinity, `RequireNodeLocal` is
added. This strongly implies that node-local services are of `ClusterIP` type
and point to DaemonSets, both reasonable assumptions to make.

Last, but not least: the current assumption is that service and DaemonSet reside
in the same namespace. There is currently no easy way to provide a global
service spanning multiple or all namespaces. Clients would need to be aware not
just of the service name, but also of the standard namespace it lives in. This
might be revisited once external services are implemented: administrators could
create "aliases" to the real service in all other namespaces that need it.

The new API, with changes highlighted in bold, would look like
<pre>
type ServiceAffinity string

const (
    // ServiceAffinityClientIP is the Client IP based.
    ServiceAffinityClientIP ServiceAffinity = "ClientIP"

    // ServiceAffinityNone - no session affinity.
    ServiceAffinityNone ServiceAffinity = "None"

    <b>// ServiceAffinityRequireNodeLocal - only talk to a pod on the same
    // node, typically managed by a Daemonset.
    ServiceAffinityRequireNodeLocal ServiceAffinity = "RequireNodeLocal"</b>
)

type ServiceSpec struct {
    Ports []ServicePort

    // If not specified, the associated Endpoints object is not automatically managed
    Selector map[string]string

    // "", a real IP, or "None".  If not specified, this is default allocated.  If "None", this Service is not load-balanced
    ClusterIP string

    // Type determines how the service will be exposed.  Valid options: ClusterIP, NodePort, LoadBalancer
    Type ServiceType

    // Only applies if clusterIP != "None"
    ExternalIPs []string

    <b>// Optional, used to maintain session affinity.
    // Supports "ClientIP", "RequireNodeLocal" and "None". The second makes
    // most sense with `Type: ClusterIP`.</b>
    SessionAffinity ServiceAffinity

    // Only applies to type=LoadBalancer
    LoadBalancerIP string
    LoadBalancerSourceRanges []string
</pre>

### `kube-proxy` changes

The most fundamental modification necessary for the proxy is becoming aware of
the node's IP address(es). Until now, it has behaved identically across all
nodes in the cluster: e.g., in iptables mode, it creates the same set of rules
everywhere, so node identity has never been required. Discovering the right
address might be tricky, especially if starting from a node name and/or
considering differences between cloud providers — or between those and bare
metal environment. Because the proxy is currently not linked against cloud
providers and in order to avoid doing so, a compromise would be to match the
list of endpoints in a service against the addresses of all the network
interfaces on the node. Should there be concerns about the cost of looking up
all the interfaces on a regular basis, a simple one-entry cache (an `<address,
interface>` pair, with the full search as a fallback) might obviate a lot of the
extra overhead.

#### iptables kernel mode

In the existing implementation, `kube-proxy` keeps track of all N endpoints for
a service. On each node, it creates `2*N` iptables rules to pick a random
endpoint when opening a new connection: one rule for the probabilistic selection
and another one for the actual `DNAT` packet rewriting. For node-local services,
it will require only one rule or even zero, if it keeps track of the actual pod
liveness. The end result is less overhead over a conventional service, before
even taking into account network latency.

For use cases like `kube2iam`, which requires intercepting traffic meant for the
special `169.254.169.254` metadata IP address, perhaps `ClusterIP` should be
allowed to live outside of the service IP range. For security reasons, this
behaviour would require explicit activation from the administrator (an API
server or proxy flag).

#### ipvs kernel mode

Issue [#17470](https://github.com/kubernetes/kubernetes/issues/17470) calls for
a newer ipvs mode. This would work similarly to iptables mode.

#### proxy userland mode

In this mode, iptables rules currently redirect the VIP to a local port owned by
the proxy, which then forwards traffic back and forth between the client and the
server pod. For node-local services, there is no need to make traffic go through
the proxy, since there are no endpoint candidates to choose from. Thus, the code
will be very similar to, if not shared with, the iptables case.

### n-to-1 forwarding

For scenarios like `loasd`, it would be useful to let multiple services point to
the same daemon. Setting multiple `DNAT` rules translating from different VIPs
to the same local port is trivial, but in most cases the daemon will want to
find out which VIP a connection was meant for. Possible options:

1. Have DaemonSets open N different ports, with node-local services pointing to
   a specific Daemonset and port
1. Rely on in-band, protocol-specific information, such as the `Host:` header
   for HTTP traffic (which might make use of HTTPS more complicated)
1. Forward traffic to addresses in the `127.0.0.0/8` network, while at the same
   time annotating node-local services with their individual loopback address.
   This also requires:
   - keeping track of the _VIP_->_loopback_ translations in the controller
     manager
   - having all the node-local daemons talk to the API server to watch services
   - enabling the routing of external traffic destined for the loopback address
   range, by running `sysctl -w net.ipv4.conf.eth0.route_localnet=1` on each
   node

   and happens to be crazy (if it can be made to work reliably at all).

## Implementation plan

### API

1. Change ServiceAffinity and comments
1. Add defaults
1. Add validation
1. Add tests
1. Add limits and/or access control

### kube-proxy

1. Figure host address
1. Implement iptables change
1. Implement userland change
1. Add tests
1. Implement special case code to handle, optionally, cloud provider metadata IP
interception (169.254.169.254)

### Documentation

1. Hunt all relevant places in the kubernetes and docs repositories and mention
   the new affinity
1. (Optional) Mention the new service type in DaemonSet docs and examples

## Future work

Possible improvements and extensions:

 - node-local services pointing to bare host ports, not pods, for daemons that
  are provisioned outside of Kubernetes. That would entail additional security
  measures.
 - "soft" node-local services, i.e. a `PreferNodeLocal` type that falls back to
   a random pod if none are running on the node.
 - rack-local services, if and when Kubernetes provides a standard way to
   capture node/rack topologies.

## Security considerations

Perhaps there should be additional checks or knobs to prevent access to ports in
the privileged range.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/node-local-services.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
