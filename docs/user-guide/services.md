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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/services.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Services in Kubernetes

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Services in Kubernetes](#services-in-kubernetes)
  - [Overview](#overview)
  - [Defining a service](#defining-a-service)
    - [Services without selectors](#services-without-selectors)
  - [Virtual IPs and service proxies](#virtual-ips-and-service-proxies)
  - [Multi-Port Services](#multi-port-services)
  - [Choosing your own IP address](#choosing-your-own-ip-address)
    - [Why not use round-robin DNS?](#why-not-use-round-robin-dns)
  - [Discovering services](#discovering-services)
    - [Environment variables](#environment-variables)
    - [DNS](#dns)
  - [Headless services](#headless-services)
  - [External services](#external-services)
    - [Type NodePort](#type-nodeport)
    - [Type LoadBalancer](#type-loadbalancer)
  - [Shortcomings](#shortcomings)
  - [Future work](#future-work)
  - [The gory details of virtual IPs](#the-gory-details-of-virtual-ips)
    - [Avoiding collisions](#avoiding-collisions)
    - [IPs and VIPs](#ips-and-vips)
  - [API Object](#api-object)

<!-- END MUNGE: GENERATED_TOC -->

## Overview

Kubernetes [`Pods`](pods.md) are mortal. They are born and they die, and they
are not resurrected.  [`ReplicationControllers`](replication-controller.md) in
particular create and destroy `Pods` dynamically (e.g. when scaling up or down
or when doing [rolling updates](kubectl/kubectl_rolling-update.md)).  While each `Pod` gets its own IP address, even
those IP addresses cannot be relied upon to be stable over time. This leads to
a problem: if some set of `Pods` (let's call them backends) provides
functionality to other `Pods` (let's call them frontends) inside the Kubernetes
cluster, how do those frontends find out and keep track of which backends are
in that set?

Enter `Services`.

A Kubernetes `Service` is an abstraction which defines a logical set of `Pods`
and a policy by which to access them - sometimes called a micro-service.  The
set of `Pods` targeted by a `Service` is (usually) determined by a [`Label
Selector`](labels.md#label-selectors) (see below for why you might want a
`Service` without a selector).

As an example, consider an image-processing backend which is running with 3
replicas.  Those replicas are fungible - frontends do not care which backend
they use.  While the actual `Pods` that compose the backend set may change, the
frontend clients should not need to be aware of that or keep track of the list
of backends themselves.  The `Service` abstraction enables this decoupling.

For Kubernetes-native applications, Kubernetes offers a simple `Endpoints` API
that is updated whenever the set of `Pods` in a `Service` changes.  For
non-native applications, Kubernetes offers a virtual-IP-based bridge to Services
which redirects to the backend `Pods`.

## Defining a service

A `Service` in Kubernetes is a REST object, similar to a `Pod`.  Like all of the
REST objects, a `Service` definition can be POSTed to the apiserver to create a
new instance.  For example, suppose you have a set of `Pods` that each expose
port 9376 and carry a label "app=MyApp".

```json
{
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "my-service"
    },
    "spec": {
        "selector": {
            "app": "MyApp"
        },
        "ports": [
            {
                "protocol": "TCP",
                "port": 80,
                "targetPort": 9376
            }
        ]
    }
}
```

This specification will create a new `Service` object named "my-service" which
targets TCP port 9376 on any `Pod` with the "app=MyApp" label.  This `Service`
will also be assigned an IP address (sometimes called the "cluster IP"), which
is used by the service proxies (see below).  The `Service`'s selector will be
evaluated continuously and the results will be POSTed to an `Endpoints` object
also named "my-service".

Note that a `Service` can map an incoming port to any `targetPort`.  By default
the `targetPort` will be set to the same value as the `port` field.  Perhaps
more interesting is that `targetPort` can be a string, referring to the name of
a port in the backend `Pods`.  The actual port number assigned to that name can
be different in each backend `Pod`. This offers a lot of flexibility for
deploying and evolving your `Services`.  For example, you can change the port
number that pods expose in the next version of your backend software, without
breaking clients.

Kubernetes `Services` support `TCP` and `UDP` for protocols.  The default
is `TCP`.

### Services without selectors

Services generally abstract access to Kubernetes `Pods`, but they can also
abstract other kinds of backends.  For example:

  * You want to have an external database cluster in production, but in test
    you use your own databases.
  * You want to point your service to a service in another
    [`Namespace`](namespaces.md) or on another cluster.
  * You are migrating your workload to Kubernetes and some of your backends run
    outside of Kubernetes.

In any of these scenarios you can define a service without a selector:

```json
{
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "my-service"
    },
    "spec": {
        "ports": [
            {
                "protocol": "TCP",
                "port": 80,
                "targetPort": 9376
            }
        ]
    }
}
```

Because this has no selector, the corresponding `Endpoints` object will not be
created. You can manually map the service to your own specific endpoints:

```json
{
    "kind": "Endpoints",
    "apiVersion": "v1",
    "metadata": {
        "name": "my-service"
    },
    "subsets": [
        {
            "addresses": [
                { "IP": "1.2.3.4" }
            ],
            "ports": [
                { "port": 80 }
            ]
        }
    ]
}
```

Accessing a `Service` without a selector works the same as if it had selector.
The traffic will be routed to endpoints defined by the user (`1.2.3.4:80` in
this example).

## Virtual IPs and service proxies

Every node in a Kubernetes cluster runs a `kube-proxy`.  This application
watches the Kubernetes master for the addition and removal of `Service`
and `Endpoints` objects. For each `Service` it opens a port (randomly chosen)
on the local node.  Any connections made to that port will be proxied to one of
the corresponding backend `Pods`.  Which backend to use is decided based on the
`SessionAffinity` of the `Service`.  Lastly, it installs iptables rules which
capture traffic to the `Service`'s cluster IP (which is virtual) and `Port` and
redirects that traffic to the previously described port.

The net result is that any traffic bound for the `Service` is proxied to an
appropriate backend without the clients knowing anything about Kubernetes or
`Services` or `Pods`.

![Services overview diagram](services-overview.png)

By default, the choice of backend is random.  Client-IP based session affinity
can be selected by setting `service.spec.sessionAffinity` to `"ClientIP"` (the
default is `"None"`).

As of Kubernetes 1.0, `Services` are a "layer 3" (TCP/UDP over IP) construct.  We do not
yet have a concept of "layer 7" (HTTP) services.

## Multi-Port Services

Many `Services` need to expose more than one port.  For this case, Kubernetes
supports multiple port definitions on a `Service` object.  When using multiple
ports you must give all of your ports names, so that endpoints can be
disambiguated.  For example:

```json
{
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "my-service"
    },
    "spec": {
        "selector": {
            "app": "MyApp"
        },
        "ports": [
            {
                "name": "http",
                "protocol": "TCP",
                "port": 80,
                "targetPort": 9376
            },
            {
                "name": "https",
                "protocol": "TCP",
                "port": 443,
                "targetPort": 9377
            }
        ]
    }
}
```

## Choosing your own IP address

You can specify your own cluster IP address as part of a `Service` creation
request.  To do this, set the `spec.clusterIP` field. For example, if you
already have an existing DNS entry that you wish to replace, or legacy systems
that are configured for a specific IP address and difficult to re-configure.
The IP address that a user chooses must be a valid IP address and within the
service_cluster_ip_range CIDR range that is specified by flag to the API
server.  If the IP address value is invalid, the apiserver returns a 422 HTTP
status code to indicate that the value is invalid.

### Why not use round-robin DNS?

A question that pops up every now and then is why we do all this stuff with
virtual IPs rather than just use standard round-robin DNS.  There are a few
reasons:

   * There is a long history of DNS libraries not respecting DNS TTLs and
     caching the results of name lookups.
   * Many apps do DNS lookups once and cache the results.
   * Even if apps and libraries did proper re-resolution, the load of every
     client re-resolving DNS over and over would be difficult to manage.

We try to discourage users from doing things that hurt themselves.  That said,
if enough people ask for this, we may implement it as an alternative.

## Discovering services

Kubernetes supports 2 primary modes of finding a `Service` - environment
variables and DNS.

### Environment variables

When a `Pod` is run on a `Node`, the kubelet adds a set of environment variables
for each active `Service`.  It supports both [Docker links
compatible](https://docs.docker.com/userguide/dockerlinks/) variables (see
[makeLinkVariables](http://releases.k8s.io/HEAD/pkg/kubelet/envvars/envvars.go#L49))
and simpler `{SVCNAME}_SERVICE_HOST` and `{SVCNAME}_SERVICE_PORT` variables,
where the Service name is upper-cased and dashes are converted to underscores.

For example, the Service "redis-master" which exposes TCP port 6379 and has been
allocated cluster IP address 10.0.0.11 produces the following environment
variables:

```bash
REDIS_MASTER_SERVICE_HOST=10.0.0.11
REDIS_MASTER_SERVICE_PORT=6379
REDIS_MASTER_PORT=tcp://10.0.0.11:6379
REDIS_MASTER_PORT_6379_TCP=tcp://10.0.0.11:6379
REDIS_MASTER_PORT_6379_TCP_PROTO=tcp
REDIS_MASTER_PORT_6379_TCP_PORT=6379
REDIS_MASTER_PORT_6379_TCP_ADDR=10.0.0.11
```

*This does imply an ordering requirement* - any `Service` that a `Pod` wants to
access must be created before the `Pod` itself, or else the environment
variables will not be populated.  DNS does not have this restriction.

### DNS

An optional (though strongly recommended) [cluster
add-on](http://releases.k8s.io/HEAD/cluster/addons/README.md) is a DNS server.  The
DNS server watches the Kubernetes API for new `Services` and creates a set of
DNS records for each.  If DNS has been enabled throughout the cluster then all
`Pods` should be able to do name resolution of `Services` automatically.

For example, if you have a `Service` called "my-service" in Kubernetes
`Namespace` "my-ns" a DNS record for "my-service.my-ns" is created.  `Pods`
which exist in the "my-ns" `Namespace` should be able to find it by simply doing
a name lookup for "my-service".  `Pods` which exist in other `Namespaces` must
qualify the name as "my-service.my-ns".  The result of these name lookups is the
cluster IP.

Kubernetes also supports DNS SRV (service) records for named ports.  If the
"my-service.my-ns" `Service` has a port named "http" with protocol `TCP`, you
can do a DNS SRV query for "_http._tcp.my-service.my-ns" to discover the port
number for "http".

## Headless services

Sometimes you don't need or want load-balancing and a single service IP.  In
this case, you can create "headless" services by specifying `"None"` for the
cluster IP (`spec.clusterIP`).

For such `Services`, a cluster IP is not allocated. DNS is configured to return
multiple A records (addresses) for the `Service` name, which point directly to
the `Pods` backing the `Service`.  Additionally, the kube proxy does not handle
these services and there is no load balancing or proxying done by the platform
for them.  The endpoints controller will still create `Endpoints` records in
the API.

This option allows developers to reduce coupling to the Kubernetes system, if
they desire, but leaves them freedom to do discovery in their own way.
Applications can still use a self-registration pattern and adapters for other
discovery systems could easily be built upon this API.

## External services

For some parts of your application (e.g. frontends) you may want to expose a
Service onto an external (outside of your cluster, maybe public internet) IP
address.  Kubernetes supports two ways of doing this: `NodePort`s and
`LoadBalancer`s.

Every `Service` has a `type` field which defines how the `Service` can be
accessed.  Valid values for this field are:

   * `ClusterIP`: use a cluster-internal IP only - this is the default and is
     discussed above
   * `NodePort`: use a cluster IP, but also expose the service on a port on each
     node of the cluster (the same port on each node)
   * `LoadBalancer`: use a ClusterIP and a NodePort, but also ask the cloud
     provider for a load balancer which forwards to the `Service`

Note that while `NodePort`s can be TCP or UDP, `LoadBalancer`s only support TCP
as of Kubernetes 1.0.

### Type NodePort

If you set the `type` field to `"NodePort"`, the Kubernetes master will
allocate a port from a flag-configured range (default: 30000-32767), and each
node will proxy that port (the same port number on every node) into your `Service`.
That port will be reported in your `Service`'s `spec.ports[*].nodePort` field.

If you want a specific port number, you can specify a value in the `nodePort`
field, and the system will allocate you that port or else the API transaction
will fail.  The value you specify must be in the configured range for node
ports.

This gives developers the freedom to set up their own load balancers, to
configure cloud environments that are not fully supported by Kubernetes, or
even to just expose one or more nodes' IPs directly.

### Type LoadBalancer

On cloud providers which support external load balancers, setting the `type`
field to `"LoadBalancer"` will provision a load balancer for your `Service`.
The actual creation of the load balancer happens asynchronously, and
information about the provisioned balancer will be published in the `Service`'s
`status.loadBalancer` field.  For example:

```json
{
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "my-service"
    },
    "spec": {
        "selector": {
            "app": "MyApp"
        },
        "ports": [
            {
                "protocol": "TCP",
                "port": 80,
                "targetPort": 9376,
                "nodePort": 30061
            }
        ],
        "clusterIP": "10.0.171.239",
        "type": "LoadBalancer"
    },
    "status": {
        "loadBalancer": {
            "ingress": [
                {
                    "ip": "146.148.47.155"
                }
            ]
        }
    }
}
```

Traffic from the external load balancer will be directed at the backend `Pods`,
though exactly how that works depends on the cloud provider.

## Shortcomings

We expect that using iptables and userspace proxies for VIPs will work at
small to medium scale, but may not scale to very large clusters with thousands
of Services.  See [the original design proposal for
portals](https://github.com/GoogleCloudPlatform/kubernetes/issues/1107) for more
details.

Using the kube-proxy obscures the source-IP of a packet accessing a `Service`.
This makes some kinds of firewalling impossible.

LoadBalancers only support TCP, not UDP.

The `Type` field is designed as nested functionality - each level adds to the
previous.  This is not strictly required on all cloud providers (e.g. Google Compute Engine does
not need to allocate a `NodePort` to make `LoadBalancer` work, but AWS does)
but the current API requires it.

## Future work

In the future we envision that the proxy policy can become more nuanced than
simple round robin balancing, for example master-elected or sharded.  We also
envision that some `Services` will have "real" load balancers, in which case the
VIP will simply transport the packets there.

There's a
[proposal](https://github.com/GoogleCloudPlatform/kubernetes/issues/3760) to
eliminate userspace proxying in favor of doing it all in iptables.  This should
perform better and fix the source-IP obfuscation, though is less flexible than
arbitrary userspace code.

We intend to have first-class support for L7 (HTTP) `Services`.

We intend to have more flexible ingress modes for `Services` which encompass
the current `ClusterIP`, `NodePort`, and `LoadBalancer` modes and more.

## The gory details of virtual IPs

The previous information should be sufficient for many people who just want to
use `Services`.  However, there is a lot going on behind the scenes that may be
worth understanding.

### Avoiding collisions

One of the primary philosophies of Kubernetes is that users should not be
exposed to situations that could cause their actions to fail through no fault
of their own.  In this situation, we are looking at network ports - users
should not have to choose a port number if that choice might collide with
another user.  That is an isolation failure.

In order to allow users to choose a port number for their `Services`, we must
ensure that no two `Services` can collide.  We do that by allocating each
`Service` its own IP address.

To ensure each service receives a unique IP, an internal allocator atomically
updates a global allocation map in etcd prior to each service. The map object
must exist in the registry for services to get IPs, otherwise creations will
fail with a message indicating an IP could not be allocated. A background
controller is responsible for creating that map (to migrate from older versions
of Kubernetes that used in memory locking) as well as checking for invalid
assignments due to administrator intervention and cleaning up any IPs
that were allocated but which no service currently uses.

### IPs and VIPs

Unlike `Pod` IP addresses, which actually route to a fixed destination,
`Service` IPs are not actually answered by a single host.  Instead, we use
`iptables` (packet processing logic in Linux) to define virtual IP addresses
which are transparently redirected as needed.  When clients connect to the
VIP, their traffic is automatically transported to an appropriate endpoint.
The environment variables and DNS for `Services` are actually populated in
terms of the `Service`'s VIP and port.

As an example, consider the image processing application described above.
When the backend `Service` is created, the Kubernetes master assigns a virtual
IP address, for example 10.0.0.1.  Assuming the `Service` port is 1234, the
`Service` is observed by all of the `kube-proxy` instances in the cluster.
When a proxy sees a new `Service`, it opens a new random port, establishes an
iptables redirect from the VIP to this new port, and starts accepting
connections on it.

When a client connects to the VIP the iptables rule kicks in, and redirects
the packets to the `Service proxy`'s own port.  The `Service proxy` chooses a
backend, and starts proxying traffic from the client to the backend.

This means that `Service` owners can choose any port they want without risk of
collision.  Clients can simply connect to an IP and port, without being aware
of which `Pods` they are actually accessing.

![Services detailed diagram](services-detail.png)

## API Object

Service is a top-level resource in the kubernetes REST API. More details about the
API object can be found at: [Service API
object](https://htmlpreview.github.io/?https://github.com/GoogleCloudPlatform/kubernetes/HEAD/docs/api-reference/definitions.html#_v1_service).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/services.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
