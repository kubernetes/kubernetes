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
[here](http://releases.k8s.io/release-1.0/docs/proposals/loadbalancing.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# L7 Loadbalancing and ingress

## Abstract

This document proposes a basic L7 loadbalancing api for kubernetes 1.1. Expect hand-waving for anything not in scope.

## Motivation

The proposed api changes are largely motivated by the confusion surrounding kubernetes ingress and loadbalancing witnessed on [user forums](#appendix). Users tend to reason about traffic that originates outside the cluster at the application layer. The current kubernetes api lacks L7 support alltogether, and has a very opionated networking model. This leads to an impedance mismatch high enough to drive them to port brokering.

## Use Cases

It goes without saying that loadbalancers play a vital role in maintaining uptime and client connectivity. Kubernetes needs an easy, HA solution for routing traffic from an external network into the cluster using application layer intelligence. Master proxy is not an acceptable solution.

## Scope

This proposal is limited to the kubernetes 1.1 release. L7 loadbalancing is a vast topic and we cannot commit to an api without iterating on community feedback.

__What's in scope for 1.1__
* A single application using tls termination or raw http should not need more than 1 loadbalancer to dispatch multiple url endpoints to different backend services. Eg:

  ```
  www.example.com -> |terminate ssl| 178.91.123.132 -> / foo    s1
                                                       / bar    s2
                                                       / foobar s3
  ```

* A singleton service can continue to ask for L4 loadbalancing. Whether the request results in a new ip, or a new port on an existing ip, is left to the loadbalancer controller. The GCE controller for example, would give you a new ip.

__What's not in scope for 1.1 (though it might be discussed below for context)__
* Multiple hostnames per l7 loadbalancer.
* Multiple certs per l7 loadbalancer.
* Exposing loadbalancing algorithms/persistence options through the api.
* A plugin model for multiple loadbalancers per cluster (i.e in 1.1, all paths will get claimed by the one loadbalancer controller the admin configures).
* Users requesting loadbalancers (i.e a claims model).

## Proposal

The following proposal goes beyond the scope of 1.1 just so we don't make myopic decisions about the API. The tl;dr proposal for 1.1 is to define an IngressPoint resource type and write an L7 GCE load balancer controller that satisfies ingress requests. A draft of the L7 controller is available [here](https://github.com/kubernetes/kubernetes/pull/12825).

__Terminology__:
* **Ingress points**: A resource representing a collection of inbound connections from the external network that would be satisfied by a load balancer. Similar to GCE's `UrlMaps` or OpenShift's `Routes`.
* **Claim**: A resource that represents a claim on an IP address.
* **Load Balancer Controller**: a controller capable of fulfilling all Ingress paths for a given class of claims.

__Example__:
* Cluster is bootstrapped with kube-controller-manager --loadbalancer-controllers=(cloud=gce),haproxy
* User creates svc1 and svc2
* User creates claim1 to get a public ip.
  GCEController:
    - receives watch notification for `Class: cloud` object.
    - allocates cloud resources for a new L7
    - spins off a goroutine that watches services for that one claim and updates the L7
* User creates claim2.
  Haproxy Controller:
    - receives watch notification for `Class: haproxy` object
    - creates a haproxy pod
    - haproxy pod watches services for that one claim and reconfigures haproxy
* We end up with 2 ips:
    - 172.12.14.111:80 serving foo and bar at L7 through a GCE L7
    - 172.11.14.143:8080 switching between foobar and baz with different certs using sni at L4, through a pod in the cluster.

```
                Cloud                                  HaProxy pod
                  ^                                         ^
                  |                                         |
            --------------------                     -------------------------
           | GCE Controller pod |                   | Ha proxy Controller pod |
            --------------------                     -------------------------
                  |                                         |
                  v                                         v
            {Name: claim1,                          {Name: claim2,
             Class: cloud,                           Class: bare-metal,
             Ip: 172.11.14.111}                      Ip: 172.11.14.143:8080}
                 /       \                              /        \
             Path1      Path2                       Path3        Path4
             type: l7   type: l7                    type: l4     type: l4
             foo.com    bar.com                     foobar.com   baz.com
             /a - svc1  /prod - svc2                cert: c1     cert: c2
             /b - svc1                              svc4         svc5
```

#### Ingress Path

This is an example ingress path. It encapsulates:
* L7 proxying information (i.e map these urls to these backends)
* Security configuration (i.e tlsMode, secret)
* Metadata needed for more advanced loadbalancing (e.g session persistence)

See [draft pr](https://github.com/kubernetes/kubernetes/pull/12825) for an actual description of the resource. See [alternatives](#alternatives) for why this deviates from existing implementations in OpenShift or GCE:

```yaml
apiVersion: v1
kind: IngressPoint
metadata:
  name: l7ingress
  type: l7
spec:
  host: example
  tlsMode: Termination
  secret: foosecret
  pathMap:
    "foo.example.com":
    - url: "/foo/*"
      service:
        name: foosvc
        namespace: default
        port: 80
    - url: "/bar/*"
      service:
        name: barsvc
        namespace: default
        port: 80
    "bar.example.com"
    - url: "/foobar/*"
        service:
          name: foobarsvc
          namespace: default
          port: 80
status:
  # include host here when we have DNS figured out.
  address: 104.101.11.39
```

Put more rigidly:

```go
// IngressPoint represents a point for ingress traffic.
type IngressPoint struct {
	TypeMeta
	ObjectMeta
	Spec IngressPointSpec
	Status IngressPointStatus
}

// IngressPointSpec describes the ingressPoint the user wishes to exist.
type IngressPointSpec struct {
	Host    string
	IngressPoint map[string][]Path
}

// IngressPointStatus describes the current state of an ingressPoint.
type IngressPointStatus struct {
	Address string
}

// ServiceRef is a reference to a single service:port.
type ServiceRef struct {
	Name      string
	Namespace string
	Port      int64
}

// Path connects a url path regex to a service:port.
type Path struct {
	Url     string
	Service ServiceRef
}
```


#### Claims

In the context of 1.1 a cluster will only have one l7 loadbalancer controller that claims all IngressPoints so we don't need claims. In the larger scheme of things, we might still not need claims, however they solve the impedance mismatch problem (why do I need a loadbalancer to expose my service?) because that is how you get an ip. If you want to expose your service, you need a public ip for it. No matter where you're running you can get one by creating a claim. If you want something to use an ip you already have, create a claim for it.

The downside is the user needs to create another resource. We can make this easier by defaulting claim-less ingress paths to a new claim.

#### Load Balancer Controllers

The duties of a loadbalancer controller are pretty straight forward: fulfill Ingress Paths. With claims, they have the additional duty of provisioning a loadbalancer backed public-ip. For all intents and purposes a loadbalancer controller is a black box that subscribes to updates to the IngressPoint resource. The administrator needs to bootstrap the cluster with loadbalancer controllers.

## Alternatives

__Loadbalancer controllers__

1. Extend the current model: With todays model, we have a service type of loadbalancer. Cloud providers implement an interface that a (single) service controller uses to create loadbalancers in response to new loadbalanced service requests. We could extend this model by just putting more information into the service spec, and adding an l7 interface to the cloud provider.

  Shortcomings of coupling loadbalancers with the cloud provider:
  * To integrate a bare metal loadbalancer users need to implement a sham cloud provider. We have evidence that this is confusing, shoehorns people into an existing interface, and forces them to dig through service controller to debug a faulty implementation.
  * Multiple bare metal loadbalancer controllers should be allowed to exist in the same cluster, along with a cloud loadbalancer controller.
  * The loadblancer interface needs to be more-or-less hermetic. This inhibits sharing of cluster level resources (i.e I only need 1 Instance Group for N loadbalancers in gce, but the createTCPLoadBalancer method needs to function without global cloud knowledge because I don't know how this works for AWS, etc).

  Shortcomings of putting more into the service definition:
  * Users view a website as being composed of a suite of services (see irc logs), so forcing them to declare path mappings in the service spec is going to be counter intuitive.
  * It is possible to run multiple websites on the same loadbalancer ip, embedding everything in the service makes this harder.

2. Admin provisions single/static set of loadbalancers: Cloud loadbalancers have seemingly artificial limitations that prevent just one loadbalancer from being generally useful for all Ingress Paths. Going this way would mean disallowing certain Ingress paths based on the cloud provider.

__Ingress paths__

1. Routes: A pathmap is conceptually inline with how people view the problem, and allows us to perform atomic switch over for blue green deployments. We can still expose a "route like" update interface so users don't have send the entire pathmap to change a single url endpoint.

2. GCEs UrlMap: This resource is way too confusing. It contains a list of host rules each of which can point to a list of path matchers, each of which has a list of path regexes. Users end up bearing the brunt of this complexity.

3. Load balancer resource: This resource would encapsulate all the information in claims and ingress paths. It will get unmanageable in deployments where a single global loadbalancer is claiming all ingress paths. This also doesn't provide a clean abstraction between acquirigin an IP and loadbalancing (the user impedance mismatch problem), while in theory we can have a non-loadbalancer fulfill a claim for an ip, or prepopulate a claim with a known ip just to have services reference it.

__Claims__

1. Have a global loadbalancer claim Ingress paths till it can claim no more: If we decide that users don't get to request new public ips, we could just dissolve the claim into the ingress paths and make the loadbalancer controllers smarter.
  * Users want to use an ip already in DNS for the service
  * The controllers need to become smarter of cloudprovider limitations like only using a single cert per loadbalancer ip
  * Detecting when a single loadbalancer is saturated can be hard

### Issues

What follows is a digest of some of the problems brought up in various github issues. Expect these to make it into the proposal above in due time.

#### How does this work at L4?

People choose L4 for a few different reasons:
- Cost
- Latency
- Throughput of loadbalancer is the bottleneck
- Cloud provider limitations
- Want SSL passthrough or SNI switch for multi-certs
- Need client IP and cannot set loadbalancer proxy as default gateway

Point being, we can't reliably infer all these use case with Service.Type=LoadBalancer or the absence of a UrlMap in the IngressPoint (because they might want L7 for session persistence). Instead, anything serving ingress traffic needs an ingress path (which is why it isn't called urlMap). An L4 ingress path might look like:

```yaml
apiVersion: v1
kind: IngressPoint
metadata:
  name: l4ingress
  type: network
spec:
  host: www.example.com
  tlsMode: Termination
  secret: foosecret
status:
  loadBalancer:
    ingress:
    - ip: 104.1..
```

type: network with a pathmap is a validation error.
This adds a slight burden on users, since what was previously a one line change to expose a service now requires a new resource.


### Why not use a single loadbalancer for all Ingress paths?

Use cases for not having all ingress paths join an existing load balancer:
* User wants only foo.com and bar.com backed by a single ip
* Internal services get bare metal loadbalancing, user facing services are backed by a cloud loadbalancer
* User already has a specific ip in DNS, and wants to expose a loadbalanced website on it
* A user wants to isolate their host from decryption of other hosts saturating cpu on a single bare metal loadbalancer (this isn't really a problem for cloud) by explicitly asking for a new one.)

### How does the ingress path choose a loadbalancer to join?

For 1.1, there will only be one loadbalancer controller that claims all ingress paths. Whether it creates a new loadbalancer or reuses an existing one is upto the controller. For example, the GCE controller will create 1 loadbalancer per ingress path, if it has `type: network`, this will be an l4, if it has `type: application` it will be an l7 with rules to allow proxying to different backends based on the given pathmap.

However we need to provide a way for users to specify which class of loadbalancers to join, and which particular loadbalancer (in the case of multiple loadbalancer per class). Note that we can't just say each class of loadbalancers has a single loadbalancer in it, and so picking a class boils down to picking a specific loadbalancer, because of cloud provider limitations. Nor can we say picking a specific loadbalancer picks the class, because a loadbalancer might have limited capacity (i.e don't force a user to pick 172.12, which might be maxed out, when they just want class=gce).

The proposed long term solution for this problem is to reference a claim on an ip through the Ingress Path.

### How does TLS work?

Each IngressPoint specifies a single secret and a tls mode. Its upto the loadbalancer controller to handle the tls mode. For example, the gce controller will only support TLS termination for 1.1 because each IngressPoint gets a new loadbalancer, and GCE doesn't allow multiple certs per IP. With the current design a user can create the wrong type of secret for the wrong mode. They wouldn't know till they try creating the Ingress path. Validating a simple key:value secret is also hard.

Some alternatives to the proposed tls mode handling:
* Remove tlsMode and create a new type of secret(s).
* Don't use a secret, embed the TLSConfig in the IngressPoint.

Regardless of how we implement it, we should probably treat tlsmode as a request and not a directive. Even if the user provides a secret the admin may choose to not expose it, or have a policy that controls that.

### What triggers the creation of a new loadbalancer controller, should this be exposed to users?

In the current proposal, the kube-controller-manager is started up with a --loadbalancer-controllers flag that dictates which controllers to startup. This is obviously sub-optimal. A better solution would be --loadbalancer-controllers=gce-lbc-pod,haproxy-lbc-pod..., then this just boils down to creating some rcs. Taking this a step further, we could have a resource to allow dynamically adding more loadbalancer controllers to the cluster.

### How does public DNS work?

Most of this is TDB. With the current model, if someone specifies a hostname the loadbalancer controllers assume they know what they're doing. If there are 2 IngressPoints with the same hostname and url endpoints, one of them will win. This is just like the overlapping labels rc problem. Since this is ultimately a policy and security decision that admins make, having replaceable/plugin controllers with each one choosing a policy might suffice.

### What does the apiserver validate vs the loadbalancer controller?

Not all loadbalancer backends will support all IngressPoint configuration. There will be some confusion around validation errors. The apiserver will validate api constituents and nothing more (i.e that an IngressPoint points to a valid service, that the tlsmodes match up to the provided secrets etc). It's probably sufficient if the loadbalancer controllers have a way to write back their status.

### How do we expose more complex loadbalancing concepts through the api?

This is the most interesting question. We need initial discussion to reach a conclusion first. In this context "more advanced" is really not *that* advanced:
* Tls modes other than termination
* Persistent sessions
* Loadbalancing algorithms (cloud provider support at this level is limited)
* Loadbalancer health checks: are they even necessary given that pods have probes?

## Appendix

IRC conversations referenced in the motivation section [motivation section](#motivation).

https://botbot.me/freenode/google-containers/msg/46082349/
https://botbot.me/freenode/google-containers/msg/47253127/
https://botbot.me/freenode/google-containers/msg/47243869/
https://botbot.me/freenode/google-containers/msg/47236167/
https://botbot.me/freenode/google-containers/msg/46085083/
https://botbot.me/freenode/google-containers/msg/46296776/
https://botbot.me/freenode/google-containers/msg/45689787/
https://botbot.me/freenode/google-containers/msg/46696926/
https://botbot.me/freenode/google-containers/msg/47238489/




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/loadbalancing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
