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
[here](http://releases.k8s.io/release-1.3/docs/proposals/federation-lite.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Multi-AZ Clusters

## (previously nicknamed "Ubernetes-Lite")

## Introduction

Full Cluster Federation will offer sophisticated federation between multiple kubernetes
clusters, offering true high-availability, multiple provider support &
cloud-bursting, multiple region support etc.  However, many users have
expressed a desire for a "reasonably" high-available cluster, that runs in
multiple zones on GCE or availability zones in AWS, and can tolerate the failure
of a single zone without the complexity of running multiple clusters.

Multi-AZ Clusters aim to deliver exactly that functionality: to run a single
Kubernetes cluster in multiple zones.  It will attempt to make reasonable
scheduling decisions, in particular so that a replication controller's pods are
spread across zones, and it will try to be aware of constraints - for example
that a volume cannot be mounted on a node in a different zone.

Multi-AZ Clusters are deliberately limited in scope; for many advanced functions
the answer will be "use full Cluster Federation".  For example, multiple-region
support is not in scope.  Routing affinity (e.g. so that a webserver will
prefer to talk to a backend service in the same zone) is similarly not in
scope.

## Design

These are the main requirements:

1. kube-up must allow bringing up a cluster that spans multiple zones.
1. pods in a replication controller should attempt to spread across zones.
1. pods which require volumes should not be scheduled onto nodes in a different zone.
1. load-balanced services should work reasonably

### kube-up support

kube-up support for multiple zones will initially be considered
advanced/experimental functionality, so the interface is not initially going to
be particularly user-friendly.  As we design the evolution of kube-up, we will
make multiple zones better supported.

For the initial implementation, kube-up must be run multiple times, once for
each zone.  The first kube-up will take place as normal, but then for each
additional zone the user must run kube-up again, specifying
`KUBE_USE_EXISTING_MASTER=true` and `KUBE_SUBNET_CIDR=172.20.x.0/24`.  This will then
create additional nodes in a different zone, but will register them with the
existing master.

### Zone spreading

This will be implemented by modifying the existing scheduler priority function
`SelectorSpread`.  Currently this priority function aims to put pods in an RC
on different hosts, but it will be extended first to spread across zones, and
then to spread across hosts.

So that the scheduler does not need to call out to the cloud provider on every
scheduling decision, we must somehow record the zone information for each node.
The implementation of this will be described in the implementation section.

Note that zone spreading is 'best effort'; zones are just be one of the factors
in making scheduling decisions, and thus it is not guaranteed that pods will
spread evenly across zones.  However, this is likely desirable: if a zone is
overloaded or failing, we still want to schedule the requested number of pods.

### Volume affinity

Most cloud providers (at least GCE and AWS) cannot attach their persistent
volumes across zones.  Thus when a pod is being scheduled, if there is a volume
attached, that will dictate the zone.  This will be implemented using a new
scheduler predicate (a hard constraint): `VolumeZonePredicate`.

When `VolumeZonePredicate` observes a pod scheduling request that includes a
volume, if that volume is zone-specific, `VolumeZonePredicate` will exclude any
nodes not in that zone.

Again, to avoid the scheduler calling out to the cloud provider, this will rely
on information attached to the volumes.  This means that this will only support
PersistentVolumeClaims, because direct mounts do not have a place to attach
zone information.  PersistentVolumes will then include zone information where
volumes are zone-specific.

### Load-balanced services should operate reasonably

For both AWS & GCE, Kubernetes creates a native cloud load-balancer for each
service of type LoadBalancer.  The native cloud load-balancers on both AWS &
GCE are region-level, and support load-balancing across instances in multiple
zones (in the same region).  For both clouds, the behaviour of the native cloud
load-balancer is reasonable in the face of failures (indeed, this is why clouds
provide load-balancing as a primitve).

For multi-AZ clusters we will therefore simply rely on the native cloud provider
load balancer behaviour, and we do not anticipate substantial code changes.

One notable shortcoming here is that load-balanced traffic still goes through
kube-proxy controlled routing, and kube-proxy does not (currently) favor
targeting a pod running on the same instance or even the same zone.  This will
likely produce a lot of unnecessary cross-zone traffic (which is likely slower
and more expensive).  This might be sufficiently low-hanging fruit that we
choose to address it in kube-proxy / multi-AZ clusters, but this can be addressed
after the initial implementation.


## Implementation

The main implementation points are:

1. how to attach zone information to Nodes and PersistentVolumes
1. how nodes get zone information
1. how volumes get zone information

### Attaching zone information

We must attach zone information to Nodes and PersistentVolumes, and possibly to
other resources in future.  There are two obvious alternatives: we can use
labels/annotations, or we can extend the schema to include the information.

For the initial implementation, we propose to use labels.  The reasoning is:

1. It is considerably easier to implement.
1. We will reserve the two labels `failure-domain.alpha.kubernetes.io/zone` and
`failure-domain.alpha.kubernetes.io/region` for the two pieces of information
we need.  By putting this under the `kubernetes.io` namespace there is no risk
of collision, and by putting it under `alpha.kubernetes.io` we clearly mark
this as an experimental feature.
1. We do not yet know whether these labels will be sufficient for all
environments, nor which entities will require zone information.  Labels give us
more flexibility here.
1. Because the labels are reserved, we can move to schema-defined fields in
future using our cross-version mapping techniques.

### Node labeling

We do not want to require an administrator to manually label nodes.  We instead
modify the kubelet to include the appropriate labels when it registers itself.
The information is easily obtained by the kubelet from the cloud provider.

### Volume labeling

As with nodes, we do not want to require an administrator to manually label
volumes.  We will create an admission controller `PersistentVolumeLabel`.
`PersistentVolumeLabel` will intercept requests to create PersistentVolumes,
and will label them appropriately by calling in to the cloud provider.

## AWS Specific Considerations

The AWS implementation here is fairly straightforward.  The AWS API is
region-wide, meaning that a single call will find instances and volumes in all
zones.  In addition, instance ids and volume ids are unique per-region (and
hence also per-zone).  I believe they are actually globally unique, but I do
not know if this is guaranteed; in any case we only need global uniqueness if
we are to span regions, which will not be supported by multi-AZ clusters (to do
that correctly requires a full Cluster Federation type approach).

## GCE Specific Considerations

The GCE implementation is more complicated than the AWS implementation because
GCE APIs are zone-scoped.  To perform an operation, we must perform one REST
call per zone and combine the results, unless we can determine in advance that
an operation references a particular zone.  For many operations, we can make
that determination, but in some cases - such as listing all instances, we must
combine results from calls in all relevant zones.

A further complexity is that GCE volume names are scoped per-zone, not
per-region.  Thus it is permitted to have two volumes both named `myvolume` in
two different GCE zones. (Instance names are currently unique per-region, and
thus are not a problem for multi-AZ clusters).

The volume scoping leads to a (small) behavioural change for multi-AZ clusters on
GCE.  If you had two volumes both named `myvolume` in two different GCE zones,
this would not be ambiguous when Kubernetes is operating only in a single zone.
But, when operating a cluster across multiple zones, `myvolume` is no longer
sufficient to specify a volume uniquely.  Worse, the fact that a volume happens
to be unambigious at a particular time is no guarantee that it will continue to
be unambigious in future, because a volume with the same name could
subsequently be created in a second zone.  While perhaps unlikely in practice,
we cannot automatically enable multi-AZ clusters for GCE users if this then causes
volume mounts to stop working.

This suggests that (at least on GCE), multi-AZ clusters must be optional (i.e.
there must be a feature-flag).  It may be that we can make this feature
semi-automatic in future, by detecting whether nodes are running in multiple
zones, but it seems likely that kube-up could instead simply set this flag.

For the initial implementation, creating volumes with identical names will
yield undefined results.  Later, we may add some way to specify the zone for a
volume (and possibly require that volumes have their zone specified when
running in multi-AZ cluster mode).  We could add a new `zone` field to the
PersistentVolume type for GCE PD volumes, or we could use a DNS-style dotted
name for the volume name (<name>.<zone>)

Initially therefore, the GCE changes will be to:

1. change kube-up to support creation of a cluster in multiple zones
1. pass a flag enabling multi-AZ clusters with kube-up
1. change the kubernetes cloud provider to iterate through relevant zones when resolving items
1. tag GCE PD volumes with the appropriate zone information


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/federation-lite.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
