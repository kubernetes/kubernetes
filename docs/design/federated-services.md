# Kubernetes Cluster Federation (previously nicknamed "Ubernetes")

## Cross-cluster Load Balancing and Service Discovery

### Requirements and System Design

### by Quinton Hoole, Dec 3 2015

## Requirements

### Discovery, Load-balancing and Failover

1. **Internal discovery and connection**: Pods/containers (running in
   a Kubernetes cluster) must be able to easily discover and connect
   to endpoints for Kubernetes services on which they depend in a
   consistent way, irrespective of whether those services exist in a
   different kubernetes cluster within the same cluster federation.
   Hence-forth referred to as "cluster-internal clients", or simply
   "internal clients".
1. **External discovery and connection**: External clients (running
   outside a Kubernetes cluster) must be able to discover and connect
   to endpoints for Kubernetes services on which they depend.
   1. **External clients predominantly speak HTTP(S)**: External
      clients are most often, but not always, web browsers, or at
      least speak HTTP(S) - notable exceptions include Enterprise
      Message Busses (Java, TLS), DNS servers (UDP),
      SIP servers and databases)
1. **Find the "best" endpoint:** Upon initial discovery and
   connection, both internal and external clients should ideally find
   "the best" endpoint if multiple eligible endpoints exist.  "Best"
   in this context implies the closest (by network topology) endpoint
   that is both operational (as defined by some positive health check)
   and not overloaded (by some published load metric).  For example:
   1. An internal client should find an endpoint which is local to its
      own cluster if one exists, in preference to one in a remote
      cluster (if both are operational and non-overloaded).
      Similarly, one in a nearby cluster (e.g. in the same zone or
      region) is preferable to one further afield.
   1. An external client (e.g. in New York City) should find an
      endpoint in a nearby cluster (e.g. U.S. East Coast) in
      preference to one further away (e.g. Japan).
1. **Easy fail-over:** If the endpoint to which a client is connected
   becomes unavailable (no network response/disconnected) or
   overloaded, the client should reconnect to a better endpoint,
   somehow.
   1. In the case where there exist one or more connection-terminating
      load balancers between the client and the serving Pod, failover
      might be completely automatic (i.e. the client's end of the
      connection remains intact, and the client is completely
      oblivious of the fail-over). This approach incurs network speed
      and cost penalties (by traversing possibly multiple load
      balancers), but requires zero smarts in clients, DNS libraries,
      recursing DNS servers etc, as the IP address of the endpoint
      remains constant over time.
   1. In a scenario where clients need to choose between multiple load
      balancer endpoints (e.g. one per cluster), multiple DNS A
      records associated with a single DNS name enable even relatively
      dumb clients to try the next IP address in the list of returned
      A records (without even necessarily re-issuing a DNS resolution
      request).  For example, all major web browsers will try all A
      records in sequence until a working one is found (TBD: justify
      this claim with details for Chrome, IE, Safari, Firefox).
   1. In a slightly more sophisticated scenario, upon disconnection, a
      smarter client might re-issue a DNS resolution query, and
      (modulo DNS record TTL's which can typically be set as low as 3
      minutes, and buggy DNS resolvers, caches and libraries which
      have been known to completely ignore TTL's), receive updated A
      records specifying a new set of IP addresses to which to
      connect.

### Portability

A Kubernetes application configuration (e.g. for a Pod, Replication
Controller, Service etc) should be able to be successfully deployed
into any Kubernetes Cluster or Federation of Clusters,
without modification. More specifically, a typical configuration
should work correctly (although possibly not optimally) across any of
the following environments:

1. A single Kubernetes Cluster on one cloud provider (e.g. Google
   Compute Engine, GCE).
1. A single Kubernetes Cluster on a different cloud provider
   (e.g. Amazon Web Services, AWS).
1. A single Kubernetes Cluster on a non-cloud, on-premise data center
1. A Federation of Kubernetes Clusters all on the same cloud provider
   (e.g. GCE).
1. A Federation of Kubernetes Clusters across multiple different cloud
   providers and/or on-premise data centers (e.g. one cluster on
   GCE/GKE, one on AWS, and one on-premise).

### Trading Portability for Optimization

It should be possible to explicitly opt out of portability across some
subset of the above environments in order to take advantage of
non-portable load balancing and DNS features of one or more
environments. More specifically, for example:

1. For HTTP(S) applications running on GCE-only Federations,
   [GCE Global L7 Load Balancers](https://cloud.google.com/compute/docs/load-balancing/http/global-forwarding-rules)
   should be usable. These provide single, static global IP addresses
   which load balance and fail over globally (i.e. across both regions
   and zones). These allow for really dumb clients, but they only
   work on GCE, and only for HTTP(S) traffic.
1. For non-HTTP(S) applications running on GCE-only Federations within
   a single region,
   [GCE L4 Network Load Balancers](https://cloud.google.com/compute/docs/load-balancing/network/)
   should be usable. These provide TCP (i.e. both HTTP/S and
   non-HTTP/S) load balancing and failover, but only on GCE, and only
   within a single region.
   [Google Cloud DNS](https://cloud.google.com/dns) can be used to
   route traffic between regions (and between different cloud
   providers and on-premise clusters, as it's plain DNS, IP only).
1. For applications running on AWS-only Federations,
   [AWS Elastic Load Balancers (ELB's)](https://aws.amazon.com/elasticloadbalancing/details/)
   should be usable. These provide both L7 (HTTP(S)) and L4 load
   balancing, but only within a single region, and only on AWS
   ([AWS Route 53 DNS service](https://aws.amazon.com/route53/) can be
   used to load balance and fail over across multiple regions, and is
   also capable of resolving to non-AWS endpoints).

## Component Cloud Services

Cross-cluster Federated load balancing is built on top of the following:

1. [GCE Global L7 Load Balancers](https://cloud.google.com/compute/docs/load-balancing/http/global-forwarding-rules)
   provide single, static global IP addresses which load balance and
   fail over globally (i.e. across both regions and zones). These
   allow for really dumb clients, but they only work on GCE, and only
   for HTTP(S) traffic.
1. [GCE L4 Network Load Balancers](https://cloud.google.com/compute/docs/load-balancing/network/)
   provide both HTTP(S) and non-HTTP(S) load balancing and failover,
   but only on GCE, and only within a single region.
1. [AWS Elastic Load Balancers (ELB's)](https://aws.amazon.com/elasticloadbalancing/details/)
   provide both L7 (HTTP(S)) and L4 load balancing, but only within a
   single region, and only on AWS.
1. [Google Cloud DNS](https://cloud.google.com/dns) (or any other
   programmable DNS service, like
   [CloudFlare](http://www.cloudflare.com) can be used to route
   traffic between regions (and between different cloud providers and
   on-premise clusters, as it's plain DNS, IP only). Google Cloud DNS
   doesn't provide any built-in geo-DNS, latency-based routing, health
   checking, weighted round robin or other advanced capabilities.
   It's plain old DNS. We would need to build all the aforementioned
   on top of it. It can provide internal DNS services (i.e. serve RFC
   1918 addresses).
   1. [AWS Route 53 DNS service](https://aws.amazon.com/route53/) can
   be used to load balance and fail over across regions, and is also
   capable of routing to non-AWS endpoints). It provides built-in
   geo-DNS, latency-based routing, health checking, weighted
   round robin and optional tight integration with some other
   AWS services (e.g. Elastic Load Balancers).
1. Kubernetes L4 Service Load Balancing: This provides both a
   [virtual cluster-local](http://kubernetes.io/v1.1/docs/user-guide/services.html#virtual-ips-and-service-proxies)
   and a
   [real externally routable](http://kubernetes.io/v1.1/docs/user-guide/services.html#type-loadbalancer)
   service IP which is load-balanced (currently simple round-robin)
   across the healthy pods comprising a service within a single
   Kubernetes cluster.
1. [Kubernetes Ingress](http://kubernetes.io/v1.1/docs/user-guide/ingress.html):
A generic wrapper around cloud-provided L4 and L7 load balancing services, and
roll-your-own load balancers run in pods, e.g. HA Proxy.

## Cluster Federation API

The Cluster Federation API for load balancing should be compatible with the equivalent
Kubernetes API, to ease porting of clients between Kubernetes and
federations of Kubernetes clusters.
Further details below.

## Common Client Behavior

To be useful, our load balancing solution needs to work properly with real
client applications. There are a few different classes of those...

### Browsers

These are the most common external clients. These are all well-written. See below.

### Well-written clients

1. Do a DNS resolution every time they connect.
1. Don't cache beyond TTL (although a small percentage of the DNS
   servers on which they rely might).
1. Do try multiple A records (in order) to connect.
1. (in an ideal world) Do use SRV records rather than hard-coded port numbers.

Examples:

+  all common browsers (except for SRV records)
+  ...

### Dumb clients

1. Don't do a DNS resolution every time they connect (or do cache beyond the
TTL).
1. Do try multiple A records

Examples:

+  ...

### Dumber clients

1. Only do a DNS lookup once on startup.
1. Only try the first returned DNS A record.

Examples:

+  ...

### Dumbest clients

1. Never do a DNS lookup - are pre-configured with a single (or possibly
multiple) fixed server IP(s). Nothing else matters.

## Architecture and Implementation

### General Control Plane Architecture

Each cluster hosts one or more Cluster Federation master components (Federation API
servers, controller managers with leader election, and etcd quorum members. This
is documented in more detail in a separate design doc:
[Kubernetes and Cluster Federation Control Plane Resilience](https://docs.google.com/document/d/1jGcUVg9HDqQZdcgcFYlWMXXdZsplDdY6w3ZGJbU7lAw/edit#).

In the description below, assume that 'n' clusters, named 'cluster-1'...
'cluster-n' have been registered against a Cluster Federation "federation-1",
each with their own set of Kubernetes API endpoints,so,
"[http://endpoint-1.cluster-1](http://endpoint-1.cluster-1),
[http://endpoint-2.cluster-1](http://endpoint-2.cluster-1)
... [http://endpoint-m.cluster-n](http://endpoint-m.cluster-n) .

### Federated Services

Federated Services are pretty straight-forward.  They're comprised of multiple
equivalent underlying Kubernetes Services, each with their own external
endpoint, and a load balancing mechanism across them. Let's work through how
exactly that works in practice.

Our user creates the following Federated Service (against a Federation
API endpoint):

    $ kubectl create -f my-service.yaml --context="federation-1"

where service.yaml contains the following:

    kind: Service
    metadata:
      labels:
        run: my-service
      name: my-service
      namespace: my-namespace
    spec:
      ports:
      - port: 2379
        protocol: TCP
        targetPort: 2379
        name: client
      - port: 2380
        protocol: TCP
        targetPort: 2380
        name: peer
      selector:
        run: my-service
      type: LoadBalancer

The Cluster Federation control system in turn creates one equivalent service (identical config to the above)
in each of the underlying Kubernetes clusters, each of which results in
something like this:

    $ kubectl get -o yaml --context="cluster-1" service my-service

    apiVersion: v1
    kind: Service
    metadata:
      creationTimestamp: 2015-11-25T23:35:25Z
      labels:
        run: my-service
      name: my-service
      namespace: my-namespace
      resourceVersion: "147365"
      selfLink: /api/v1/namespaces/my-namespace/services/my-service
      uid: 33bfc927-93cd-11e5-a38c-42010af00002
    spec:
      clusterIP: 10.0.153.185
      ports:
      - name: client
        nodePort: 31333
        port: 2379
        protocol: TCP
        targetPort: 2379
      - name: peer
        nodePort: 31086
        port: 2380
        protocol: TCP
        targetPort: 2380
      selector:
        run: my-service
      sessionAffinity: None
      type: LoadBalancer
    status:
      loadBalancer:
        ingress:
        - ip: 104.197.117.10

Similar services are created in `cluster-2` and `cluster-3`, each of which are
allocated their own `spec.clusterIP`, and `status.loadBalancer.ingress.ip`.

In the Cluster Federation `federation-1`, the resulting federated service looks as follows:

    $ kubectl get -o yaml --context="federation-1" service my-service

    apiVersion: v1
    kind: Service
    metadata:
      creationTimestamp: 2015-11-25T23:35:23Z
      labels:
        run: my-service
      name: my-service
      namespace: my-namespace
      resourceVersion: "157333"
      selfLink: /api/v1/namespaces/my-namespace/services/my-service
      uid: 33bfc927-93cd-11e5-a38c-42010af00007
    spec:
      clusterIP:
      ports:
      - name: client
        nodePort: 31333
        port: 2379
        protocol: TCP
        targetPort: 2379
      - name: peer
        nodePort: 31086
        port: 2380
        protocol: TCP
        targetPort: 2380
      selector:
        run: my-service
      sessionAffinity: None
      type: LoadBalancer
    status:
      loadBalancer:
        ingress:
        - hostname: my-service.my-namespace.my-federation.my-domain.com

Note that the federated service:

1. Is API-compatible with a vanilla Kubernetes service.
1. has no clusterIP (as it is cluster-independent)
1. has a federation-wide load balancer hostname

In addition to the set of underlying Kubernetes services (one per cluster)
described above, the Cluster Federation control system has also created a DNS name (e.g. on
[Google Cloud DNS](https://cloud.google.com/dns) or
[AWS Route 53](https://aws.amazon.com/route53/), depending on configuration)
which provides load balancing across all of those services. For example, in a
very basic configuration:

    $ dig +noall +answer my-service.my-namespace.my-federation.my-domain.com
    my-service.my-namespace.my-federation.my-domain.com 180 IN	A 104.197.117.10
    my-service.my-namespace.my-federation.my-domain.com 180 IN	A 104.197.74.77
    my-service.my-namespace.my-federation.my-domain.com 180 IN	A 104.197.38.157

Each of the above IP addresses (which are just the external load balancer
ingress IP's of each cluster service) is of course load balanced across the pods
comprising the service in each cluster.

In a more sophisticated configuration (e.g. on GCE or GKE), the Cluster
Federation control system
automatically creates a
[GCE Global L7 Load Balancer](https://cloud.google.com/compute/docs/load-balancing/http/global-forwarding-rules)
which exposes a single, globally load-balanced IP:

    $ dig +noall +answer my-service.my-namespace.my-federation.my-domain.com
    my-service.my-namespace.my-federation.my-domain.com 180 IN	A 107.194.17.44

Optionally, the Cluster Federation control system also configures the local DNS servers (SkyDNS)
in each Kubernetes cluster to preferentially return the local
clusterIP for the service in that cluster, with other clusters'
external service IP's (or a global load-balanced IP) also configured
for failover purposes:

    $ dig +noall +answer my-service.my-namespace.my-federation.my-domain.com
    my-service.my-namespace.my-federation.my-domain.com 180 IN	A 10.0.153.185
    my-service.my-namespace.my-federation.my-domain.com 180 IN	A 104.197.74.77
    my-service.my-namespace.my-federation.my-domain.com 180 IN	A 104.197.38.157

If Cluster Federation Global Service Health Checking is enabled, multiple service health
checkers running across the federated clusters collaborate to monitor the health
of the service endpoints, and automatically remove unhealthy endpoints from the
DNS record (e.g. a majority quorum is required to vote a service endpoint
unhealthy, to avoid false positives due to individual health checker network
isolation).

### Federated Replication Controllers

So far we have a federated service defined, with a resolvable load balancer
hostname by which clients can reach it, but no pods serving traffic directed
there. So now we need a Federated Replication Controller. These are also fairly
straight-forward, being comprised of multiple underlying Kubernetes Replication
Controllers which do the hard work of keeping the desired number of Pod replicas
alive in each Kubernetes cluster.

    $ kubectl create -f my-service-rc.yaml --context="federation-1"

where `my-service-rc.yaml` contains the following:

    kind: ReplicationController
    metadata:
      labels:
        run: my-service
      name: my-service
      namespace: my-namespace
    spec:
      replicas: 6
      selector:
        run: my-service
      template:
        metadata:
          labels:
            run: my-service
        spec:
          containers:
            image: gcr.io/google_samples/my-service:v1
            name: my-service
            ports:
            - containerPort: 2379
              protocol: TCP
            - containerPort: 2380
              protocol: TCP

The Cluster Federation control system in turn creates one equivalent replication controller
(identical config to the above, except for the replica count) in each
of the underlying Kubernetes clusters, each of which results in
something like this:

    $ ./kubectl get -o yaml rc my-service --context="cluster-1"
    kind: ReplicationController
    metadata:
      creationTimestamp: 2015-12-02T23:00:47Z
      labels:
        run: my-service
      name: my-service
      namespace: my-namespace
      selfLink: /api/v1/namespaces/my-namespace/replicationcontrollers/my-service
      uid: 86542109-9948-11e5-a38c-42010af00002
    spec:
      replicas: 2
      selector:
        run: my-service
      template:
        metadata:
          labels:
            run: my-service
        spec:
          containers:
            image: gcr.io/google_samples/my-service:v1
            name: my-service
            ports:
            - containerPort: 2379
              protocol: TCP
            - containerPort: 2380
              protocol: TCP
            resources: {}
          dnsPolicy: ClusterFirst
          restartPolicy: Always
    status:
      replicas: 2

The exact number of replicas created in each underlying cluster will of course
depend on what scheduling policy is in force. In the above example, the
scheduler created an equal number of replicas (2) in each of the three
underlying clusters, to make up the total of 6 replicas required. To handle
entire cluster failures, various approaches are possible, including:
1. **simple overprovisioning**, such that sufficient replicas remain even if a
   cluster fails. This wastes some resources, but is simple and reliable.
2. **pod autoscaling**, where the replication controller in each
      cluster automatically and autonomously increases the number of
      replicas in its cluster in response to the additional traffic
      diverted from the failed cluster. This saves resources and is relatively
      simple, but there is some delay in the autoscaling.
3. **federated replica migration**, where the Cluster Federation
   control system detects the cluster failure and automatically
   increases the replica count in the remainaing clusters to make up
   for the lost replicas in the failed cluster. This does not seem to
   offer any benefits relative to pod autoscaling above, and is
   arguably more complex to implement, but we note it here as a
   possibility.

### Implementation Details

The implementation approach and architecture is very similar to Kubernetes, so
if you're familiar with how Kubernetes works, none of what follows will be
surprising. One additional design driver not present in Kubernetes is that
the Cluster Federation control system aims to be resilient to individual cluster and availability zone
failures. So the control plane spans multiple clusters. More specifically:

+ Cluster Federation runs it's own distinct set of API servers (typically one
   or more per underlying Kubernetes cluster).  These are completely
   distinct from the Kubernetes API servers for each of the underlying
   clusters.
+  Cluster Federation runs it's own distinct quorum-based metadata store (etcd,
   by default). Approximately 1 quorum member runs in each underlying
   cluster ("approximately" because we aim for an odd number of quorum
   members, and typically don't want more than 5 quorum members, even
   if we have a larger number of federated clusters, so 2 clusters->3
   quorum members, 3->3, 4->3, 5->5, 6->5, 7->5 etc).

Cluster Controllers in the Federation control system watch against the
Federation API server/etcd
state, and apply changes to the underlying kubernetes clusters accordingly. They
also have the anti-entropy mechanism for reconciling Cluster Federation "desired desired"
state against kubernetes "actual desired" state.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/federated-services.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
