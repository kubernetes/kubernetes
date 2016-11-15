## Abstract

Real Kubernetes clusters have a variety of volumes which differ widely in
size, iops performance, retention policy, and other characteristics.  A
mechanism is needed to enable administrators to describe the taxonomy of these
volumes, and for users to make claims on these volumes based on their
attributes within this taxonomy.

A label selector mechanism is proposed to enable flexible selection of volumes
by persistent volume claims.

## Motivation

Currently, users of persistent volumes have the ability to make claims on
those volumes based on some criteria such as the access modes the volume
supports and minimum resources offered by a volume.  In an organization, there
are often more complex requirements for the storage volumes needed by
different groups of users.  A mechanism is needed to model these different
types of volumes and to allow users to select those different types without
being intimately familiar with their underlying characteristics.

As an example, many cloud providers offer a range of performance
characteristics for storage, with higher performing storage being more
expensive.  Cluster administrators want the ability to:

1.  Invent a taxonomy of logical storage classes using the attributes
    important to them
2.  Allow users to make claims on volumes using these attributes

## Constraints and Assumptions

The proposed design should:

1.  Deal with manually-created volumes
2.  Not necessarily require users to know or understand the differences between
    volumes (ie, Kubernetes should not dictate any particular set of
    characteristics to administrators to think in terms of)

We will focus **only** on the barest mechanisms to describe and implement
label selectors in this proposal.  We will address the following topics in
future proposals:

1.  An extension resource or third party resource for storage classes
1.  Dynamically provisioning new volumes for based on storage class

## Use Cases

1.  As a user, I want to be able to make a claim on a persistent volume by
    specifying a label selector as well as the currently available attributes

### Use Case: Taxonomy of Persistent Volumes

Kubernetes offers volume types for a variety of storage systems.  Within each
of those storage systems, there are numerous ways in which volume instances
may differ from one another: iops performance, retention policy, etc.
Administrators of real clusters typically need to manage a variety of
different volumes with different characteristics for different groups of
users.

Kubernetes should make it possible for administrators to flexibly model the
taxonomy of volumes in their clusters and to label volumes with their storage
class.  This capability must be optional and fully backward-compatible with
the existing API.

Let's look at an example.  This example is *purely fictitious* and the
taxonomies presented here are not a suggestion of any sort.  In the case of
AWS EBS there are four different types of volume (in ascending order of cost):

1.  Cold HDD
2.  Throughput optimized HDD
3.  General purpose SSD
4.  Provisioned IOPS SSD

Currently, there is no way to distinguish between a group of 4 PVs where each
volume is of one of these different types.  Administrators need the ability to
distinguish between instances of these types.  An administrator might decide
to think of these volumes as follows:

1.  Cold HDD - `tin`
2.  Throughput optimized HDD - `bronze`
3.  General purpose SSD - `silver`
4.  Provisioned IOPS SSD - `gold`

This is not the only dimension that EBS volumes can differ in.  Let's simplify
things and imagine that AWS has two availability zones, `east` and `west`. Our
administrators want to differentiate between volumes of the same type in these
two zones, so they create a taxonomy of volumes like so:

1.  `tin-west`
2.  `tin-east`
3.  `bronze-west`
4.  `bronze-east`
5.  `silver-west`
6.  `silver-east`
7.  `gold-west`
8.  `gold-east`

Another administrator of the same cluster might label things differently,
choosing to focus on the business role of volumes.  Say that the data
warehouse department is the sole consumer of the cold HDD type, and the DB as
a service offering is the sole consumer of provisioned IOPS volumes.  The
administrator might decide on the following taxonomy of volumes:

1.  `warehouse-east`
2.  `warehouse-west`
3.  `dbaas-east`
4.  `dbaas-west`

There are any number of ways an administrator may choose to distinguish
between volumes.  Labels are used in Kubernetes to express the user-defined
properties of API objects and are a good fit to express this information for
volumes.  In the examples above, administrators might differentiate between
the classes of volumes using the labels `business-unit`, `volume-type`, or
`region`.

Label selectors are used through the Kubernetes API to describe relationships
between API objects using flexible, user-defined criteria.  It makes sense to
use the same mechanism with persistent volumes and storage claims to provide
the same functionality for these API objects.

## Proposed Design

We propose that:

1.  A new field called `Selector` be added to the `PersistentVolumeClaimSpec`
    type
2.  The persistent volume controller be modified to account for this selector
    when determining the volume to bind to a claim

### Persistent Volume Selector

Label selectors are used throughout the API to allow users to express
relationships in a flexible manner.  The problem of selecting a volume to
match a claim fits perfectly within this metaphor.  Adding a label selector to
`PersistentVolumeClaimSpec` will allow users to label their volumes with
criteria important to them and select volumes based on these criteria.

```go
// PersistentVolumeClaimSpec describes the common attributes of storage devices
// and allows a Source for provider-specific attributes
type PersistentVolumeClaimSpec struct {
    // Contains the types of access modes required
    AccessModes []PersistentVolumeAccessMode `json:"accessModes,omitempty"`
    // Selector is a selector which must be true for the claim to bind to a volume
    Selector *unversioned.Selector `json:"selector,omitempty"`
    // Resources represents the minimum resources required
    Resources ResourceRequirements `json:"resources,omitempty"`
    // VolumeName is the binding reference to the PersistentVolume backing this claim
    VolumeName string `json:"volumeName,omitempty"`
}
```

### Labeling volumes

Volumes can already be labeled:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ebs-pv-1
  labels:
    ebs-volume-type: iops
    aws-availability-zone: us-east-1
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  awsElasticBlockStore:
    volumeID: vol-12345
    fsType: xfs
```

### Controller Changes

At the time of this writing, the various controllers for persistent volumes
are in the process of being refactored into a single controller (see
[kubernetes/24331](https://github.com/kubernetes/kubernetes/pull/24331)).

The resulting controller should be modified to use the new
`selector` field to match a claim to a volume.  In order to
match to a volume, all criteria must be satisfied; ie, if a label selector is
specified on a claim, a volume must match both the label selector and any
specified access modes and resource requirements to be considered a match.

## Examples

Let's take a look at a few examples, revisiting the taxonomy of EBS volumes and regions:

Volumes of the different types might be labeled as follows:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ebs-pv-west
  labels:
    ebs-volume-type: iops-ssd
    aws-availability-zone: us-west-1
spec:
  capacity:
    storage: 150Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  awsElasticBlockStore:
    volumeID: vol-23456
    fsType: xfs

apiVersion: v1
kind: PersistentVolume
metadata:
  name: ebs-pv-east
  labels:
    ebs-volume-type: gp-ssd
    aws-availability-zone: us-east-1
spec:
  capacity:
    storage: 150Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  awsElasticBlockStore:
    volumeID: vol-34567
    fsType: xfs
```

...claims on these volumes would look like:

```yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: ebs-claim-west
spec:
  accessModes:
    - ReadWriteMany 
  resources:
    requests:
      storage: 1Gi
  selector:
    matchLabels:
      ebs-volume-type: iops-ssd
      aws-availability-zone: us-west-1

kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: ebs-claim-east
spec:
  accessModes:
    - ReadWriteMany 
  resources:
    requests:
      storage: 1Gi
  selector:
    matchLabels:
      ebs-volume-type: gp-ssd
      aws-availability-zone: us-east-1
```



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volume-selectors.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
