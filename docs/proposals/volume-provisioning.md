<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

Real Kubernetes clusters have a variety of volumes which differ widely in
size, iops performance, retention policy, and other characteristics.
Administrators need a way to dynamically provision volumes of these different
types to automatically meet user demand.

A new mechanism called 'storage classes' is proposed to provide this
capability.

## Motivation

In Kubernetes 1.2, an alpha form of limited dynamic provisioning was added
that allows a single volume type to be provisioned in clouds that offer
special volume types.

In Kubernetes 1.3, a label selector was added to persistent volume claims to
allow administrators to create a taxonomy of volumes based on the
characteristics important to them, and to allow users to make claims on those
volumes based on those characteristics.  This allows flexibility when claiming
existing volumes; the same flexibility is needed when dynamically provisioning
volumes.

After gaining experience with dynamic provisioning after the 1.2 release, we
want to create a more flexible feature that allows configuration of how
different storage classes are provisioned and supports provisioning multiple
types of volumes within a single cloud.

### Out-of-tree provisioners

One of our goals is to enable administrators to create out-of-tree
provisioners, that is, provisioners whose code does not live in the Kubernetes
project.  Our experience since the 1.2 release with dynamic provisioning has
shown that it is impossible to anticipate every aspect and manner of
provisioning that administrators will want to perform.  The proposed design
should not prevent future work to allow out-of-tree provisioners.

## Design

This design represents the minimally viable changes required to provision based on storage classe configuration.  Additional incremental features may be added as a separte effort.

We propose that:

1.  For the base impelementation storage class and volume selectors are mutually exclusive.

2.  An api object will be incubated in extensions/v1beta1 named `storage` to hold the a `StorageClass`
    API resource. Each StorageClass object contains parameters required by the provisioner to provision volumes of that class.  These parameters are opaque to the user.

3.  `PersistentVolume.Spec.Class` attribute is added to volumes. This attribute
    is optional and specifies which `StorageClass` instance represents
    storage characteristics of a particular PV.

    During incubation, `Class` is an annotation and not
    actual attribute.

4.  `PersistentVolume` instances do not require labels by the provisioner.

5.  `PersistentVolumeClaim.Spec.Class` attribute is added to claims. This
    attribute specifies that only a volume with equal
    `PersistentVolume.Spec.Class` value can satisfy a claim.

    During incubation, `Class` is just an annotation and not
    actual attribute.

6.  The existing provisioner plugin implementations be modified to accept
    parameters as specified via `StorageClass`.

7.  The persistent volume controller modified to invoke provisioners using `StorageClass` configuration and bind claims with `PersistentVolumeClaim.Spec.Class` to volumes with equivilant `PersistentVolume.Spec.Class`

8.  The existing alpha dynamic provisioning feature be phased out in the
    next release.

### Controller workflow for provisioning volumes

1.  When a new claim is submitted, the controller attempts to find an existing
    volume that will fulfill the claim.

    1. If the claim has non-empty `claim.Spec.Class`, only PVs with the same
        `pv.Spec.Class` are considered.

    2. If the claim has empty `claim.Spec.Class`, all existing PVs are
        considered.

    All "considered" volumes are evaluated and the
    smallest matching volume is bound to the claim.

2.  If no volume is found for the claim and `claim.Spec.Class` is not set or is
    empty string dynamic provisioning is disabled.

3.  If `claim.Spec.Class` is set the controller tries to find instance of StorageClass with this name.  If no
    such StorageClass is found, the controller goes back to step 1. and
    periodically retries finding a matching volume or storage class again until
    a match is found. The claim is `Pending` during this period.

4.  With StorageClass instance, the controller finds volume plugin specified by
    StorageClass.Provisioner.

5.  All provisioners are in-tree; they implement an interface called
    `ProvisionableVolumePlugin`, which has a method called `NewProvisioner`
    that returns a new provisioner.

6.  The controller calls volume plugin `Provision` with Parameters from the `StorageClass` configuration object.

7.  If `Provision` returns an error, the controller generates an event on the
    claim and goes back to step 1., i.e. it will retry provisioning periodically

8.  If `Provision` returns no error, the controller creates the returned
    `api.PersistentVolume`, fills its `Class` attribute with `claim.Spec.Class`
    and makes it already bound to the claim

  1.  If the create operation for the `api.PersistentVolume` fails, it is
      retried

  2.  If the create operation does not succeed in reasonable time, the
      controller attempts to delete the provisioned volume and creates an event
      on the claim

Existing behavior is un-changed for claims that do not specify `claim.Spec.Class`.

### `StorageClass` API

A new API group should hold the API for storage classes, following the pattern
of autoscaling, metrics, etc.  To allow for future storage-related APIs, we
should call this new API group `storage` and incubate in extensions/v1beta1.

Storage classes will be represented by an API object called `StorageClass`:

```go
package storage

// StorageClass describes the parameters for a class of storage for
// which PersistentVolumes can be dynamically provisioned.
//
// StorageClasses are non-namespaced; the name of the storage class
// according to etcd is in ObjectMeta.Name.
type StorageClass struct {
  unversioned.TypeMeta `json:",inline"`
  ObjectMeta           `json:"metadata,omitempty"`

  // Provisioner indicates the type of the provisioner.
  Provisioner string `json:"provisioner,omitempty"`

  // Parameters for dynamic volume provisioner.
  Parameters map[string]string `json:"parameters,omitempty"`
}

```

`PersistentVolumeClaimSpec` and `PersistentVolumeSpec` both get Class attribute
(the existing annotation is used during incubation):

```go
type PersistentVolumeClaimSpec struct {
    // Name of requested storage class. If non-empty, only PVs with this
    // pv.Spec.Class will be considered for binding and if no such PV is
    // available, StorageClass with this name will be used to dynamically
    // provision the volume.
    Class string
...
}

type PersistentVolumeSpec struct {
    // Name of StorageClass instance that this volume belongs to.
    Class string
...
}
```

Storage classes are natural to think of as a global resource, since they:

1.  Align with PersistentVolumes, which are a global resource
2.  Are administrator controlled

### Provisioning configuration

With the scheme outlined above the provisioner creates PVs using parameters specified in the `StorageClass` object.

### Provisioner interface changes

`struct volume.VolumeOptions` (containing parameters for a provisioner plugin)
will be extended to contain StorageClass.Parameters.

The existing provisioner implementations will be modified to accept the StorageClass configuration object.

### PV Controller Changes

The persistent volume controller will be modified to implement the new
workflow described in this proposal.  The changes will be limited to the
`provisionClaimOperation` method, which is responsible for invoking the
provisioner and to favor existing volumes before provisioning a new one.

## Examples

### AWS provisioners with distinct QoS

This example shows two storage classes, "aws-fast" and "aws-slow".

```
apiVersion: v1
kind: StorageClass
metadata:
  name: aws-fast
provisioner: kubernetes.io/aws-ebs
parameters:
   zone: us-east-1b
   type: ssd


apiVersion: v1
kind: StorageClass
metadata:
  name: aws-slow
provisioner: kubernetes.io/aws-ebs
parameters:
   zone: us-east-1b
   type: spinning
```

# Additional Implementation Details

0. Annotation `volume.alpha.kubernetes.io/storage-class` is used instead of `claim.Spec.Class` and `volume.Spec.Class` during incubation.

1. `claim.Spec.Selector` and `claim.Spec.Class` are mutually exclusive. User can either match existing volumes with `Selector` XOR match existing volumes with `Class` and get dynamic provisioning by using `Class`. This simplifies initial PR and also provisioners.

# Cloud Providers

Since the `volume.alpha.kubernetes.io/storage-class` is in use a `StorageClass` must be defined to support provisioning.  No default is assumed as before.



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volume-provisioning.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
