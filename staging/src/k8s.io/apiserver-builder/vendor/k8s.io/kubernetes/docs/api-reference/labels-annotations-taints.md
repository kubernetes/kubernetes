# Well-Known Labels, Annotations and Taints

Kubernetes reserves all labels and annotations in the kubernetes.io namespace.  This document describes
the well-known kubernetes.io labels and annotations.

This document serves both as a reference to the values, and as a coordination point for assigning values.

**Table of contents:**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Well-Known Labels, Annotations and Taints](#well-known-labels-annotations-and-taints)
  - [beta.kubernetes.io/arch](#betakubernetesioarch)
  - [beta.kubernetes.io/os](#betakubernetesioos)
  - [kubernetes.io/hostname](#kubernetesiohostname)
  - [beta.kubernetes.io/instance-type](#betakubernetesioinstance-type)
  - [failure-domain.beta.kubernetes.io/region](#failure-domainbetakubernetesioregion)
  - [failure-domain.beta.kubernetes.io/zone](#failure-domainbetakubernetesiozone)

<!-- END MUNGE: GENERATED_TOC -->


## beta.kubernetes.io/arch

Example: `beta.kubernetes.io/arch=amd64`

Used on: Node

Kubelet populates this with `runtime.GOARCH` as defined by Go.  This can be handy if you are mixing arm and x86 nodes,
for example.

## beta.kubernetes.io/os

Example: `beta.kubernetes.io/os=linux`

Used on: Node

Kubelet populates this with `runtime.GOOS` as defined by Go.  This can be handy if you are mixing operating systems
in your cluster (although currently Linux is the only OS supported by kubernetes).

## kubernetes.io/hostname

Example: `kubernetes.io/hostname=ip-172-20-114-199.ec2.internal`

Used on: Node

Kubelet populates this with the hostname.  Note that the hostname can be changed from the "actual" hostname
by passing the `--hostname-override` flag to kubelet.

## beta.kubernetes.io/instance-type

Example: `beta.kubernetes.io/instance-type=m3.medium`

Used on: Node

Kubelet populates this with the instance type as defined by the `cloudprovider`.  It will not be set if
not using a cloudprovider.  This can be handy if you want to target certain workloads to certain instance
types, but typically you want to rely on the kubernetes scheduler to perform resource-based scheduling,
and you should aim to schedule based on properties rather than on instance types (e.g. require a GPU, instead
of requiring a `g2.2xlarge`)


## failure-domain.beta.kubernetes.io/region

See [failure-domain.beta.kubernetes.io/zone](#failure-domainbetakubernetesiozone)

## failure-domain.beta.kubernetes.io/zone

Example:

`failure-domain.beta.kubernetes.io/region=us-east-1`

`failure-domain.beta.kubernetes.io/zone=us-east-1c`

Used on: Node, PersistentVolume

On the Node: Kubelet populates this with the zone information as defined by the `cloudprovider`.  It will not be set if
not using a `cloudprovider`, but you should consider setting it on the nodes if it makes sense in your topology.

On the PersistentVolume: The `PersistentVolumeLabel` admission controller will automatically add zone labels to PersistentVolumes,
on GCE and AWS.

Kubernetes will automatically spread the pods in a replication controller or service across nodes in a single-zone
cluster (to reduce the impact of failures.) With multiple-zone clusters, this spreading behaviour is extended
across zones (to reduce the impact of zone failures.) This is achieved via SelectorSpreadPriority.

This is a best-effort placement, and so if the zones in your cluster are heterogeneous (e.g. different numbers of nodes,
different types of nodes, or different pod resource requirements), this might prevent equal spreading of
your pods across zones. If desired, you can use homogenous zones (same number and types of nodes) to reduce
the probability of unequal spreading.

The scheduler (via the VolumeZonePredicate predicate) will also ensure that pods that claim a given volume
are only placed into the same zone as that volume, as volumes cannot be attached across zones.


The actual values of zone and region don't matter, and nor is the meaning of the hierarchy rigidly defined.  The expectation
is that failures of nodes in different zones should be uncorrelated unless the entire region has failed.  For example,
zones should typically avoid sharing a single network switch.  The exact mapping depends on your particular
infrastructure - a three-rack installation will choose a very different setup to a multi-datacenter configuration.

If `PersistentVolumeLabel` does not support automatic labeling of your PersistentVolumes, you should consider
adding the labels manually (or adding support to `PersistentVolumeLabel`), if you want the scheduler to prevent
pods from mounting volumes in a different zone.  If your infrastructure doesn't have this constraint, you don't
need to add the zone labels to the volumes at all.





<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-reference/labels-annotations-taints.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
