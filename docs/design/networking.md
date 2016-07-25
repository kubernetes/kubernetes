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
[here](http://releases.k8s.io/release-1.3/docs/design/networking.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Networking

There are 4 distinct networking problems to solve:

1. Highly-coupled container-to-container communications
2. Pod-to-Pod communications
3. Pod-to-Service communications
4. External-to-internal communications

## Model and motivation

Kubernetes deviates from the default Docker networking model (though as of
Docker 1.8 their network plugins are getting closer). The goal is for each pod
to have an IP in a flat shared networking namespace that has full communication
with other physical computers and containers across the network. IP-per-pod
creates a clean, backward-compatible model where pods can be treated much like
VMs or physical hosts from the perspectives of port allocation, networking,
naming, service discovery, load balancing, application configuration, and
migration.

Dynamic port allocation, on the other hand, requires supporting both static
ports (e.g., for externally accessible services) and dynamically allocated
ports, requires partitioning centrally allocated and locally acquired dynamic
ports, complicates scheduling (since ports are a scarce resource), is
inconvenient for users, complicates application configuration, is plagued by
port conflicts and reuse and exhaustion, requires non-standard approaches to
naming (e.g. consul or etcd rather than DNS), requires proxies and/or
redirection for programs using standard naming/addressing mechanisms (e.g. web
browsers), requires watching and cache invalidation for address/port changes
for instances in addition to watching group membership changes, and obstructs
container/pod migration (e.g. using CRIU). NAT introduces additional complexity
by fragmenting the addressing space, which breaks self-registration mechanisms,
among other problems.

## Container to container

All containers within a pod behave as if they are on the same host with regard
to networking. They can all reach each other’s ports on localhost.  This offers
simplicity (static ports know a priori), security (ports bound to localhost
are visible within the pod but never outside it), and performance. This also
reduces friction for applications moving from the world of uncontainerized apps
on physical or virtual hosts. People running application stacks together on
the same host have already figured out how to make ports not conflict and have
arranged for clients to find them.

The approach does reduce isolation between containers within a pod &mdash;
ports could conflict, and there can be no container-private ports, but these
seem to be relatively minor issues with plausible future workarounds. Besides,
the premise of pods is that containers within a pod share some resources
(volumes, cpu, ram, etc.) and therefore expect and tolerate reduced isolation.
Additionally, the user can control what containers belong to the same pod
whereas, in general, they don't control what pods land together on a host.

## Pod to pod

Because every pod gets a "real" (not machine-private) IP address, pods can
communicate without proxies or translations. The pod can use well-known port
numbers and can avoid the use of higher-level service discovery systems like
DNS-SD, Consul, or Etcd.

When any container calls ioctl(SIOCGIFADDR) (get the address of an interface),
it sees the same IP that any peer container would see them coming from &mdash;
each pod has its own IP address that other pods can know. By making IP addresses
and ports the same both inside and outside the pods, we create a NAT-less, flat
address space. Running "ip addr show" should work as expected. This would enable
all existing naming/discovery mechanisms to work out of the box, including
self-registration mechanisms and applications that distribute IP addresses. We
should be optimizing for inter-pod network communication. Within a pod,
containers are more likely to use communication through volumes (e.g., tmpfs) or
IPC.

This is different from the standard Docker model. In that mode, each container
gets an IP in the 172-dot space and would only see that 172-dot address from
SIOCGIFADDR. If these containers connect to another container the peer would see
the connect coming from a different IP than the container itself knows. In short
&mdash; you can never self-register anything from a container, because a
container can not be reached on its private IP.

An alternative we considered was an additional layer of addressing: pod-centric
IP per container. Each container would have its own local IP address, visible
only within that pod. This would perhaps make it easier for containerized
applications to move from physical/virtual hosts to pods, but would be more
complex to implement (e.g., requiring a bridge per pod, split-horizon/VP DNS)
and to reason about, due to the additional layer of address translation, and
would break self-registration and IP distribution mechanisms.

Like Docker, ports can still be published to the host node's interface(s), but
the need for this is radically diminished.

## Implementation

For the Google Compute Engine cluster configuration scripts, we use [advanced
routing rules](https://developers.google.com/compute/docs/networking#routing)
and ip-forwarding-enabled VMs so that each VM has an extra 256 IP addresses that
get routed to it.  This is in addition to the 'main' IP address assigned to the
VM that is NAT-ed for Internet access.  The container bridge (called `cbr0` to
differentiate it from `docker0`) is set up outside of Docker proper.

Example of GCE's advanced routing rules:

```sh
gcloud compute routes add "${NODE_NAMES[$i]}" \
  --project "${PROJECT}" \
  --destination-range "${NODE_IP_RANGES[$i]}" \
  --network "${NETWORK}" \
  --next-hop-instance "${NODE_NAMES[$i]}" \
  --next-hop-instance-zone "${ZONE}" &
```

GCE itself does not know anything about these IPs, though. This means that when
a pod tries to egress beyond GCE's project the packets must be SNAT'ed
(masqueraded) to the VM's IP, which GCE recognizes and allows.

### Other implementations

With the primary aim of providing IP-per-pod-model, other implementations exist
to serve the purpose outside of GCE.
  - [OpenVSwitch with GRE/VxLAN](../admin/ovs-networking.md)
  - [Flannel](https://github.com/coreos/flannel#flannel)
  - [L2 networks](http://blog.oddbit.com/2014/08/11/four-ways-to-connect-a-docker/)
    ("With Linux Bridge devices" section)
  - [Weave](https://github.com/zettio/weave) is yet another way to build an
    overlay network, primarily aiming at Docker integration.
  - [Calico](https://github.com/Metaswitch/calico) uses BGP to enable real
    container IPs.

## Pod to service

The [service](../user-guide/services.md) abstraction provides a way to group pods under a
common access policy (e.g. load-balanced). The implementation of this creates a
virtual IP which clients can access and which is transparently proxied to the
pods in a Service. Each node runs a kube-proxy process which programs
`iptables` rules to trap access to service IPs and redirect them to the correct
backends. This provides a highly-available load-balancing solution with low
performance overhead by balancing client traffic from a node on that same node.

## External to internal

So far the discussion has been about how to access a pod or service from within
the cluster. Accessing a pod from outside the cluster is a bit more tricky. We
want to offer highly-available, high-performance load balancing to target
Kubernetes Services. Most public cloud providers are simply not flexible enough
yet.

The way this is generally implemented is to set up external load balancers (e.g.
GCE's ForwardingRules or AWS's ELB) which target all nodes in a cluster. When
traffic arrives at a node it is recognized as being part of a particular Service
and routed to an appropriate backend Pod. This does mean that some traffic will
get double-bounced on the network. Once cloud providers have better offerings
we can take advantage of those.

## Challenges and future work

### Docker API

Right now, docker inspect doesn't show the networking configuration of the
containers, since they derive it from another container. That information should
be exposed somehow.

### External IP assignment

We want to be able to assign IP addresses externally from Docker
[#6743](https://github.com/dotcloud/docker/issues/6743) so that we don't need
to statically allocate fixed-size IP ranges to each node, so that IP addresses
can be made stable across pod infra container restarts
([#2801](https://github.com/dotcloud/docker/issues/2801)), and to facilitate
pod migration. Right now, if the pod infra container dies, all the user
containers must be stopped and restarted because the netns of the pod infra
container will change on restart, and any subsequent user container restart
will join that new netns, thereby not being able to see its peers.
Additionally, a change in IP address would encounter DNS caching/TTL problems.
External IP assignment would also simplify DNS support (see below).

### IPv6

IPv6 would be a nice option, also, but we can't depend on it yet. Docker support
is in progress: [Docker issue #2974](https://github.com/dotcloud/docker/issues/2974),
[Docker issue #6923](https://github.com/dotcloud/docker/issues/6923),
[Docker issue #6975](https://github.com/dotcloud/docker/issues/6975).
Additionally, direct ipv6 assignment to instances doesn't appear to be supported
by major cloud providers (e.g., AWS EC2, GCE) yet. We'd happily take pull
requests from people running Kubernetes on bare metal, though. :-)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/networking.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
