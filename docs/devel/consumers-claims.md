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
[here](http://releases.k8s.io/release-1.0/docs/devel/consumers-claims.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# The consumer/claim model for cluster resources

A pattern has emerged for managing certain classes of resources (e.g. storage
and networking) in the Kubernetes API.  This doc aims to document this pattern
for future reuse.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [The consumer/claim model for cluster resources](#the-consumerclaim-model-for-cluster-resources)
  - [Background](#background)
  - [Status](#status)
  - [General model](#general-model)
  - [Storage](#storage)
    - [PersistentVolumeClaim](#persistentvolumeclaim)
    - [VolumeManagers](#volumemanagers)
    - [Config](#config)
    - [Scrubbing](#scrubbing)
  - [Networking](#networking)
    - [IngressClaim](#ingressclaim)
    - [LoadBalancerManagers](#loadbalancermanagers)
    - [Config](#config)
    - [Scrubbing](#scrubbing)
  - [Future](#future)

<!-- END MUNGE: GENERATED_TOC -->

## Background

Some types of resources which are used in a cluster follow a common lifecycle:
they are provisioned (whether manually or automatically), used by some
consumer(s), eventually released by the consumer, scrubbed, and finally either
released or reused.  These resources generally map to discrete physical or
virtual _things_.

Examples:
   * Virtual storage (e.g. cloud-provider-hosted block devices)
   * Physical storage (e.g. partitions on an iSCSI server or NFS shares)
   * Virtual networking (e.g. cloud-provider-hosted load-balancers)
   * Physical networking (e.g. VIPs on hardware load-balancers)

These resources may have long lead times to provision (e.g. file a ticket to
have an admin do work), may have requirements around de-provisioning (e.g.
disks must be erased), or cost real money (e.g. for-pay cloud IaaS).  To use
these things in Kubernetes we need a way to represent the in our API, to claim
them as in-use, and to dynamically provision them (when supported).

When thinking about these resources it is useful to keep in mind the extremes of
how they might operate.  Consider:
   * A cloud-based load-balancer which can be provisioned in seconds with just
     an API call, but costs real money.
   * A bare-metal disk device for which provisioning means sending a ticket to
     your sys-admin, who eventually types some commands into a control
     application and responds to your ticket with a device ID.

## Status

This document is somewhat aspirational.  The state of the system at the time of
this writing is NOT fully aligned with this model, but we intend for the system
to trend towards it.

## General model

Both storage and networking resources can be handled with a common model.

```
                       +----------+
        +--create----> | resource |
        |              +----------+
   +----+----+               ^
   | manager |---bind------> :
   +----+----+               v
        |              +----------+
        +---watch----> |  claim   | <--+
                       +----------+    |
                             ^         |
                             |         +---- user-created
                             |         |
                       +----------+    |
                       | consumer | <--+
                       +----------+
```

In this model a `consumer` object (created by a user in the API) files a
`claim` (also created by the user) for some type of resource.  The reason for
the separation of these concepts is that they have different lifetimes - a
claim might be reused across multiple consumers over time.  Claims are abstract
by nature (see the storage and networking forms of this diagram below) and
describe what kind of resource is needed, but (generally) not the identity of
the specific resource instance.  A `Manager` entity (a process, not part of the
API) is responsible for satisfying the claim by "binding" the claim to a
specific resource.

You can sort of see this same pattern in the scheduling of `Pod`s onto `Node`s.
A user creates a `Pod`, which is a claim against resources on a `Node`, and is
bound by the scheduler.

## Storage

Storage in Kubernetes will be represented by `PersistentVolume`s.  A user can
create a `PersistentVolumeClaim` for storage and use that claim from a `Pod`.
A `VolumeManager` is responsible for binding `PersistentVolumeClaim`s to
`PersistentVolume`s.  If the `PersistentVolumeClaim` can not be satisified by
any existing `PersistentVolume`, the `VolumeManager` may provision a new one.

```
                       +------------------+
        +--create----> | PersistentVolume |
        |              +------------------+
   +----+----+               ^
   | VolMgr  |---bind------> :
   +----+----+               v
        |              +------------------------+
        +---watch----> | PersistentVolumeClaim  | <--+
                       +------------------------+    |
                                 ^                   |
                                 |         +---------+-- user-created
                                 |         |
                           +----------+    |
                           |    Pod   | <--+
                           +----------+
```

### PersistentVolumeClaim

A `PersistentVolumeClaim` does not specify which specific PersistentVolume it
wants or even which volume plugin should be used.  Instead it specifies
characteristics of the storage it needs and those characteristics are satisfied
as best is possible.  Characteristics include the minimum storage space
required and the minimum access mode required (e.g. multi-writer).

As the ecosystem of volume plugins grows, there will be an ever-increasing
number of things that could be specified about volumes.  Rather than growing
`PersistentVolumeClaim` to cover the union of every volume's parameters, users
will be able to request coarse "classes" of volume (e.g. Gold, Silver, Bronze).
How those classes map to real storage technologies is a policy set by the
cluster administrators.

### VolumeManagers

The act of binding a `PersistentVolumeClaim` to a `PersistentVolume` is the
responsibility of a process in the cluster.  A cluster may have any number of
`VolumeManager`s running, each of which handles some set of claim classes.
When a `PersistentVolumeClaim` is created that asks for "Gold" storage, the
`VolumeManager` which understands the "Gold" class will make the binding.

`VolumeManager`s can be simple `Pod`s running in the cluster, pulling
cofiguration from hardcoded values, flags, config objects, or whatever
mechanism makes sense for them.  We expect config objects to be a perfect fit
for this use case.

This very loose coupling means that an admin can define classes however they
like.  For example, "Gold" might mean GCE PersistentDisk of type SSD, "Silver"
might mean GCE PersistentDisk of type Disk, and "Bronze" might mean NFS.
Configuring the same `VolumeManager` to run for both "Gold" and "Silver" with
just a change in params is trivial, as would be adding a totally bespoke
`VolumeManager`.

Note that classes are just identifiers - there are no special names or
meanings.  One of the `VolumeManager`s must be the "default" for claims that do
not specify a class.

#### Claim binding vs pod scheduling

Currently the pod scheduler does not consider the boundness or non-boundness of
a `PersistentVolumeClaim` when scheduling a `Pod` to a `Node`.  If a Pod is
scheduled with a claim that is not yet bound, the kubelet will refuse to start
the `Pod` until the claim is bound.  In the future this should be handled
before it ever hits the kubelet.

### Config

We expect that `VolumeManager`s will need some amount of configuration, and it
will be convenient if they have similar parameters, but this should be by
convention, not by diktat.  Example parameters:
   * Max instances - limit of number of volumes allowed
   * Min free instances - how many ready-to-use volumes should be maintained
   * Max free instances - limit of number of ready-to-use volumes
   * Size distribution - desired histogram of sizes of ready-to-use volumes

In addition to these, there will be volume-type specific parameters, such as
the GCE SSD vs Disk param described above.

### Scrubbing

Volumes need to be scrubbed before they can be reused, and some storage devices
may even need to be scrubbed before they are decommissioned.  This is the
responsibility of the `VolumeManager` as it drives the lifecycle.

## Networking

TODO: flip Ingress and LB ?

Network ingress in Kubernetes will be represented by `LoadBalancer`s.  A user
can create an `IngressClaim` and reference that claim from an `Ingress` object.
A `LoadBalancerManager` is responsible for binding `IngressClaim`s to
`LoadBalancers`s.  If the `IngressClaim` can not be satisified by any existing
`LoadBalancer`, the `LoadBalancerManager` may provision a new one.

```
                       +--------------+
        +--create----> | LoadBalancer |
        |              +--------------+
   +----+----+               ^
   |  LBMgr  |---bind------> :
   +----+----+               v
        |              +---------------+
        +---watch----> | IngressClaim  | <--+
                       +---------------+    |
                                 ^          |
                                 |          +-------- user-created
                                 |          |
                           +----------+     |
                           | Ingress  | <---+
                           +----------+
```

### IngressClaim

An `IngressClaim` does not specify which specific Ingress it wants.  Instead it
specifies characteristics of the ingress it needs and those characteristics are
satisfied as best is possible.  Characteristics might include the availability
properties required or the specific IP address to use.

Like `PersistentVolumeClaim`, users will be able to request coarse "classes" of
ingress (e.g. Gold, Silver, Bronze).  How those classes map to real storage
technologies is a policy set by the cluster administrators.

### LoadBalancerManagers

The act of binding a `IngressClaim` to an `Ingress` is the responsibility of a
process in the cluster.  A cluster may have any number of
`LoadBalancerManager`s running, each of which handles some set of claim
classes.  When an `IngressClaim` is created that asks for "Gold" ingress, the
`LoadBalancerManager` which understands the "Gold" class will make the binding.

`LoadBalancerManager`s can be simple `Pod`s running in the cluster, pulling
cofiguration from hardcoded values, flags, config objects, or whatever
mechanism makes sense for them.  We expect config objects to be a perfect fit
for this use case.

This very loose coupling means that an admin can define classes however they
like.  For example, "Gold" might mean GCE Cloud Load Balancer, "Silver"
might mean HAProxy, and "Bronze" might mean a port on a single VM.
Adding a totally bespoke `LoadBalancerManager` is trivial.

Note that classes are just identifiers - there are no special names or
meanings.  One of the `LoadBalancerManager`s must be the "default" for claims that do
not specify a class.

### Config

We expect that `LoadBalancerManager`s will need some amount of configuration, and it
will be convenient if they have similar parameters, but this should be by
convention, not by diktat.  Example parameters:
   * Max instances - limit of number of balancers allowed
   * Min free instances - how many ready-to-use balancers should be maintained
   * Max free instances - limit of number of ready-to-use balancers

### Scrubbing

While it's a bit more of a stretch, even load-balancers have a conceptual
scrubbing phase.  For example, if a particular IP address was used with a DNS name, the
system might choose not to reuse that IP until the TTL of the DNS name has
expired.

## Future

We hope that this pattern and the commonalities continue to hold over time.  As
we integrate new cluster resources, we hope this model will also extend to
them.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/consumers-claims.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
