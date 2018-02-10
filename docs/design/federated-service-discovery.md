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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Service Discovery in Ubernetes
==============================

**Author**: Madhu C.S. ([madhusudancs@google.com](mailto:madhusudancs@google.com))

**Last updated**: 05/03/2016

**Status**: **Draft**|Approved|Abandoned|Obsolete

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Objective](#objective)
- [Background](#background)
- [Design](#design)
  - [Apparatus](#apparatus)
  - [Method](#method)
    - [Initial Setup](#initial-setup)
    - [Internal Clients](#internal-clients)
      - [DNS Record Creation](#dns-record-creation)
      - [Service Discovery/Lookup](#service-discoverylookup)
    - [External Clients](#external-clients)
      - [DNS Record Creation](#dns-record-creation)
      - [Service Discovery/Lookup](#service-discoverylookup)
- [Caveats](#caveats)
- [Advantages](#advantages)
  - [Push model vs Pull model for programming **`k8s-dns`**](#push-model-vs-pull-model-for-programming-k8s-dns)
- [FAQ](#faq)
- [Alternatives](#alternatives)
- [References](#references)
- [Notes](#notes)

<!-- END MUNGE: GENERATED_TOC -->

# Objective

Ubernetes extends the concept of Kubernetes Services to cluster federations.
These services,  called the **Federated Services**, should be discoverable by
both the clients running within the pods in any of
the Kubernetes clusters in the federation and by the clients running outside
the federation. Clients running within the clusters should be able to
preferentially connect to the shard of the service running within the same
cluster, if one exists. Otherwise, they should be able to connect to the shard
of the service within the same zone or the same region in that order of
preference. If no such shards exist, then they should be able to connect to any
shard of the service. In addition, clients should be able to explicitly specify
the shard of the service they want to connect to.

# Background

Kubernetes provides service discovery through two different mechanisms:

  1. Environment variables
  2. DNS

In Ubernetes, we do not intend to support service discovery through environment
variables for various reasons discussed in the [Federated (Ubernetes) Service Discovery DNS details - rough notes](https://docs.google.com/a/google.com/document/d/1Lqe8rhiLveoDRoRQHRqL5p91_yQ1M9wIT8THvbTVH4M/edit?usp=sharing)
doc. We will continue to support the mechanisms provided by Kubernetes for the
local shards of federated services, just as if they were services within a
single Kubernetes cluster, but clients will not be able to discover shards in
other clusters using this mechanism.

Service discovery for federated services will be provided through DNS. [Federated (Ubernetes) Service Discovery DNS details - rough notes](https://docs.google.com/a/google.com/document/d/1Lqe8rhiLveoDRoRQHRqL5p91_yQ1M9wIT8THvbTVH4M/edit?usp=sharing)
and [Federated Services](federated-services.md)
docs have more detailed discussions on the requirements of such a DNS-based
discovery scheme. Please refer to those docs for details and additional
background.

# Design

The overall architecture of the service discovery mechanism for federated
services in Ubernetes is shown in the diagram below.

<!-- TODO: Update the rawgit.com URL to point to kubernetes/kubernetes before merging this PR. -->
![Federated Service Discovery Architecture](https://rawgit.com/madhusudancs/kubernetes/federated-service-discovery/docs/design/federated-service-discovery-arch.svg)

## Apparatus

Each individual Kubernetes cluster runs its own DNS server, an instance of
`SkyDNS`, as a cluster addon that provides service discovery for cluster-local
services. Every container in every pod in the cluster is configured to use this
DNS server instance for domain name resolution. The rest of this doc refers to
this DNS server instance as **`K8S-DNS`**.

**`K8S-DNS`** runs an instance of `SkyDNS` along with its own `etcd` and
`kube2sky` controller in a Kubernetes pod. Each of these components run in
their own container within the pod. `kube2sky` watches the services in the
cluster and programs the etcd backing the `SkyDNS`. In addition to these
components, another container running an HTTP server, called stubzone server,
is introduced into this pod to facilitate Ubernetes service discovery. This
HTTP server exposes a single endpoint that allows programming the
[stub zones](https://github.com/skynetservices/skydns#stub-zones) in the
`etcd` for `SkyDNS`. An existing Kubernetes cluster needs to replace its
**`K8S-DNS`** pod with a pod containing this HTTP server container to join
a federation.

In addition to **`K8S-DNS`**, Ubernetes runs its own DNS instance to provide
service discovery for federated services. A single instance of this DNS server
is sufficient per federation. However, there is nothing in the design that
stops HA installations from configuring multiple instances. The rest of this
doc refers to this DNS instance as **`Federated-DNS`**.

**`Federated-DNS`** also runs within a pod in one of the Kubernetes clusters in
the federation. This pod consists of a DNS server backed by its own `etcd` and a
controller called `FederatedDNSController`.

## Method

### Initial Setup

**`Federated-DNS`** pod is started as part of the Ubernetes bootstrap process.
Whenever a Kubernetes cluster wants to join a federation, its **`KubeDNS`**
pod should be replaced with a **`K8S-DNS`** pod to enable cross-cluster
service discovery.

Cluster controller in the Ubernetes control plane watches for new clusters and
sends a POST request to the HTTP server running in the **`K8S-DNS`** pod in
each new cluster to update the stub zone information for that federation. This
HTTP server needs TLS client-side authentication for the clients to POST
requests to this server. Cluster controller POSTs the request to the path:


```
/federations/myfederation
```

With the following JSON encoded content:


```
{
  nameservers: ["108.45.90.180", "108.180.64.128"]
}
```

Federation administrators must also specify whether they want the federated
services to be visible to external clients, i.e. clients outside the cluster
federation. If they want to enable such a behavior, they should provide the
name of a programmable DNS provider such as Google Cloud DNS and the
credentials to that service. In addition, they should also specify the
registered domain name to use for the external federated service discovery. The
details of when, where and how to provide this configuration is outside the
scope of this doc and will be discussed elsewhere. This doc assumes that the
necessary DNS provider information and a registered domain name is available to
the **`Federated-DNS`** instance.

### Internal Clients

#### DNS Record Creation

Upon creation of a federated service in the Ubernetes API server, the federated
service controller in Ubernetes watching for these services creates the
corresponding service shards in the individual Kubernetes clusters as specified
in the spec. These Kubernetes services are independent of each other, but
Ubernetes has a holistic view of these services.

`FederatedDNSController` in **`Federated-DNS`** watches for these services and
updates the local `etcd` with external and cluster-local addresses for each
service shard. The external addresses are the loadbalancer IP addresses of the
service shards that constitute the federated service. `etcd` key-value pairs
for a service *myservice* in namespace *myns* and clusters *foo*, *bar* and
*baz* in the federation *myfederation* look as follows:

```JSON
Key: /myfederation/myns/myservice/foo
Value: {
    "externalIP": "104.32.64.128",
    "clusterIP": "10.0.3.64",
}
Key: /myfederation/myns/myservice/bar
Value: {
    "externalIP": "104.32.64.130",
    "clusterIP": "10.0.4.20",
}
Key: /myfederation/myns/myservice/baz
Value: {
    "externalIP": "108.120.45.12",
    "clusterIP": "10.56.62.6",
}
```

#### Service Discovery/Lookup

A client running within a pod in any of the Kubernetes clusters in the
federation looks up for a federated service by specifying the domain name
corresponding to the service. Internal clients must always specify the fully
qualified domain name (FQDN) of the federated service to disambiguate between
cluster-local and federated services. Fully qualified domain name for the
federated service described above is specified as
`myservice.myns.myfederation.svc.federation.`

All the pods within a Kubernetes cluster are configured to use **`K8S-DNS`**
for domain name resolution. When a client tries to connect to the federated
service, the client's resolver library issues a DNS lookup request to the
**`K8S-DNS`** server. `SkyDNS` running in **`K8S-DNS`** looks at the domain
name and recognizes that there is a stub zone configured for the domain name
suffix. It adds the source cluster information, i.e. the name of the cluster
in which the **`K8S-DNS`** is running, in the EDNS(0)
[[RFC 6891](https://tools.ietf.org/html/rfc6891)] section of the lookup
request and forwards the request to the nameserver configured in the stub
zone for that domain name suffix, i.e. the **`Federated-DNS`**.

Upon receiving this lookup request, **`Federated-DNS`** retrieves all the
matching records for the federated service specified in the domain name
from its `etcd`. It then looks at the source cluster information in the
EDNS(0) section of the request and obtains the 'clusterIP' for that cluster
in the retrieved records. It then obtains the 'externalIP' from all other
records, sorts these IP addresses with the 'clusterIP' at the front of the
list and returns this result to the **`K8S-DNS`** that sent the lookup
request. **`K8S-DNS`** then returns this result back to its client.

Well-behaved clients that implement the IP address selection ordering as
described in [RFC 3484](https://tools.ietf.org/html/rfc3484) first attempt
to connect to the 'clusterIP', i.e. local shard of the service, because
cluster-local IP addresses in Kubernetes today have 10/8 prefixes and
these prefixes are considered to have site-local scope according to the RFC.
Even otherwise, because the IP addresses in the DNS response are sorted to have
the cluster-local address first, the clients that don't respect
[RFC 3484](https://tools.ietf.org/html/rfc3484) will still try to connect
to the local shard of the service first, as long as they don't mangle the
order of addresses in the DNS response.

### External Clients

#### DNS Record Creation

When a federated service specifies the `ServiceType` as `LoadBalancer`, the
`FederatedDNSController` in the **`Federated-DNS`** watching for these
federated services, in addition to programming its local `etcd`, also programs
the DNS provider with the loadbalancer IP addresses of each shard of the
federated service. The DNS record for a service *myservice* in namespace
*myns* and clusters *foo*, *bar* and *baz* in the federation *myfederation*
looks as follows:

```shell
Name: myservice.myns.myfederation.svc.federation.ubernetes.io.
Host: 104.32.64.128
Host: 104.32.64.130
Host: 108.120.45.12
```

#### Service Discovery/Lookup

An external client running outside the federation of clusters looks up for the
service by specifying the fully qualified domain name of the service. This
request propagates through its recursive nameservers and ends up in the DNS
provider configured for the federation. This server being authoritative for the
domain name corresponding to the federated service, answers the DNS query by
returning the list of IP addresses for all the federated service shards.

It is important to note that the fully qualified domain names for the internal
and external clients are different. An external client must specify the
registered domain name for the federation while the internal clients are not
required to specify that portion.

# Caveats

No automatic domain name expansion by using DNS search list.

Internal clients in a Kubernetes cluster can discover services in their own
same namespace, i.e. the namespace in which their pod is running, just by just
looking up for the service name. The resolver library does automatic expansion
of domain names by appending the domain suffixes from the search list in the
containers' resolv.conf.

Federated services cannot be looked up this way. This is intentional and
by-design. We do not want to silently return the federated service IP addresses
for the service lookups unless explicitly requested. We do this to:

  1. Avoid unintentional/accidental cross-cluster connections. Not doing this might
result in a service running in one cluster to connect to a service in another
cluster running in a different region and/or a provider which might in-turn
result in both unexpected network bills and performance degradation depending
on the source and the destination cluster locations. For example, if a service
in a GKE cluster in asia-east1 silently tries to connect to a service in a
Kubernetes cluster running in AWS in us-east-1 without the client's knowledge,
it results in both unexpected performance degradation and network costs.
  2. Avoid breaking existing clients. Expecting cluster administrators to replace
the **`Kube-DNS` **pod with a **`K8S-DNS`** pod is already too much to ask for.
We do not want to place the additional burden on the existing clients to ensure
that they only connect to cluster-local services when they need that behavior.
In other words, the existing behavior is to connect to cluster-local services in
the same namespace when only the service name is specified and we want to
preserve that behavior. Existing clients should not be expected to do any extra
work when they need that behavior.
  3. Ensure that the clients understand what they are doing and have them explicitly
specify that they want to connect to a federated service which might be running
across regions and providers. Moreover, clients can always connect to a local
shard of a federated service by just specifying the service name.

# Advantages

  1. One of the main advantages of this two-level nameserver design that puts
**`Federated-DNS`** outside the path of cluster-local service discovery is that
DNS resolution latencies for cluster-local services are unaffected.
  2. Not changing anything in the current name resolution path for cluster-local
services also mean that reliability of the cluster-local service discovery is
unaffected because there are no additional hoops to jump through.
  3. No configuration changes are required in any of the Kubernetes cluster
components. For example, it is not required to change `kubelet`'s'
`--cluster-dns` flag. All the `/etc/resolv.conf`s in the system including the
ones in the containers in the pod remains unchanged. The only thing that needs
to be replaced is the `kube-dns` pod.
  4. Only cross-cluster service name resolutions need to go through another
level of name resolution in the **`Federated-DNS`** server.

## Push model vs Pull model for programming **`k8s-dns`**

The proposed push model that allows programming **`k8s-dns`** stub zone information
using an HTTP server as opposed to having a controller that keeps a watch on the
`cluster` resources in the Ubernetes API server has its pros and cons:

**Pros**

  1. Having an HTTP server that exposes an endpoint to program the stubzone information
eliminates the need to keep a watch across regions.
  2. Since the HTTP server only exposes the endpoint to program stubzones, as opposed
to exposing entire `etcd`, it limits the attack surface by providing an opening to
program just the stub zones. `SkyDNS`'s underlying store cannot be touched otherwise.

**Cons**

  1. Need to run a service to expose this HTTP server outside the clusters so that
Ubernetes can program it. That also means we need to worry about authentication and
authorization to determine who can send requests to this HTTP server.
  2. Need to restart/replace the **`kube-dns`** pod with the **`K8S-DNS`** pod when
a Kubernetes cluster wants to join a federation for the first time.

# FAQ

  1. **`kube-dns`** pods in the existing Kubernetes clusters need to be
replaced with the **`K8S-DNS`** pods. Isn't this against the principal design
goal of Ubernetes that Kubernetes components shouldn't be aware of the
federation to operate?

    No, for two reasons. This is not a core Kubernetes component, but just a
    cluster addon. And people can replace this with their own solutions. See
    next question.
  2. Why not use a different, more simpler service discovery mechanism such as
Consul or use `etcd` directly?

    To enable legacy apps. However, users are free to provide their own, more
    flexible, service discovery mechanisms for their applications. DNS-based
    service discovery enables a whole class of applications that were not written
    for Kubernetes or are not Kubernetes-aware. And that is why we want to ship
    DNS-based service discovery as the default mechanism.

# Alternatives

These are alternative designs considered:

  1. Running a DNS Proxy in front of **`K8S-DNS`**. Either run it on each node as
a DaemonSet or run in it in the **`K8S-DNS`** pod.
  2. Directly programming a Cloud DNS service such as Google Cloud DNS or Amazon
Route 53 instead of making the requests go through **`Federated-DNS`**.
  3. Programming the stub zone information in **`K8S-DNS`** at start up and not
running an HTTP server for programming it.

# References

  1. [Federated (Ubernetes) Service Discovery DNS details - rough notes](https://docs.google.com/a/google.com/document/d/1Lqe8rhiLveoDRoRQHRqL5p91_yQ1M9wIT8THvbTVH4M/edit?usp=sharing)
  2. [Service Discovery described in Federated Services doc](federated-services.md)
  3. EDNS(0) - [https://tools.ietf.org/html/rfc6891](https://tools.ietf.org/html/rfc6891)
  4. IP address selection ordering - [https://tools.ietf.org/html/rfc3484](https://tools.ietf.org/html/rfc3484)

# Notes

CNAME restrictions - [https://tools.ietf.org/html/rfc1912#section-2.4](https://tools.ietf.org/html/rfc1912#section-2.4)



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/federated-service-discovery.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
