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
project.

## Design

This design represents the minimally viable changes required to provision based on storage class configuration.  Additional incremental features may be added as a separate effort.

We propose that:

1.  Both for in-tree and out-of-tree storage provisioners, the PV created by the
    provisioners must match the PVC that led to its creations. If a provisioner
    is unable to provision such a matching PV, it reports an error to the
    user.

2.  The above point applies also to PVC label selector. If user submits a PVC
    with a label selector, the provisioner must provision a PV with matching
    labels. This directly implies that the provisioner understands meaning
    behind these labels - if user submits a claim with selector that wants
    a PV with label "region" not in "[east,west]", the provisioner must
    understand what label "region" means, what available regions are there and
    choose e.g. "north".

    In other words, provisioners should either refuse to provision a volume for
    a PVC that has a selector, or select few labels that are allowed in
    selectors (such as the "region" example above), implement necessary logic
    for their parsing, document them and refuse any selector that references
    unknown labels.

3.  An api object will be incubated in storage.k8s.io/v1beta1 to hold the a `StorageClass`
    API resource. Each StorageClass object contains parameters required by the provisioner to provision volumes of that class.  These parameters are opaque to the user.

4.  `PersistentVolume.Spec.Class` attribute is added to volumes. This attribute
    is optional and specifies which `StorageClass` instance represents
    storage characteristics of a particular PV.

    During incubation, `Class` is an annotation and not
    actual attribute.

5.  `PersistentVolume` instances do not require labels by the provisioner.

6.  `PersistentVolumeClaim.Spec.Class` attribute is added to claims. This
    attribute specifies that only a volume with equal
    `PersistentVolume.Spec.Class` value can satisfy a claim.

    During incubation, `Class` is just an annotation and not
    actual attribute.

7.  The existing provisioner plugin implementations be modified to accept
    parameters as specified via `StorageClass`.

8.  The persistent volume controller modified to invoke provisioners using `StorageClass` configuration and bind claims with `PersistentVolumeClaim.Spec.Class` to volumes with equivalent `PersistentVolume.Spec.Class`

9.  The existing alpha dynamic provisioning feature be phased out in the
    next release.

### Controller workflow for provisioning volumes

0. Kubernetes administator can configure name of a default StorageClass. This
   StorageClass instance is then used when user requests a dynamically
   provisioned volume, but does not specify a StorageClass. In other words,
   `claim.Spec.Class == ""`
   (or annotation `volume.beta.kubernetes.io/storage-class == ""`).

1.  When a new claim is submitted, the controller attempts to find an existing
    volume that will fulfill the claim.

    1. If the claim has non-empty `claim.Spec.Class`, only PVs with the same
        `pv.Spec.Class` are considered.

    2. If the claim has empty `claim.Spec.Class`, only PVs with an unset `pv.Spec.Class` are considered.

    All "considered" volumes are evaluated and the
    smallest matching volume is bound to the claim.

2.  If no volume is found for the claim and `claim.Spec.Class` is not set or is
    empty string dynamic provisioning is disabled.

3.  If `claim.Spec.Class` is set the controller tries to find instance of StorageClass with this name.  If no
    such StorageClass is found, the controller goes back to step 1. and
    periodically retries finding a matching volume or storage class again until
    a match is found. The claim is `Pending` during this period.

4.  With StorageClass instance, the controller updates the claim:
       * `claim.Annotations["volume.beta.kubernetes.io/storage-provisioner"] = storageClass.Provisioner`

* **In-tree provisioning**

   The controller tries to find an internal volume plugin referenced by
   `storageClass.Provisioner`. If it is found:

  5.  The internal provisioner implements interface`ProvisionableVolumePlugin`,
      which has a method called `NewProvisioner` that returns a new provisioner.

  6.  The controller calls volume plugin `Provision` with Parameters
      from the `StorageClass` configuration object.

  7.  If `Provision` returns an error, the controller generates an event on the
      claim and goes back to step 1., i.e. it will retry provisioning
      periodically.

  8.  If `Provision` returns no error, the controller creates the returned
      `api.PersistentVolume`, fills its `Class` attribute with `claim.Spec.Class`
      and makes it already bound to the claim

    1.  If the create operation for the `api.PersistentVolume` fails, it is
        retried

    2.  If the create operation does not succeed in reasonable time, the
        controller attempts to delete the provisioned volume and creates an event
        on the claim

Existing behavior is unchanged for claims that do not specify
`claim.Spec.Class`.

* **Out of tree provisioning**

  Following step 4. above, the controller tries to find internal plugin for the
  `StorageClass`. If it is not found, it does not do anything, it just
  periodically goes to step 1., i.e. tries to find available matching PV.

  The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
  "SHOULD NOT", "RECOMMENDED",  "MAY", and "OPTIONAL" in this document are to be
  interpreted as described in RFC 2119.

  External provisioner must have these features:

  * It MUST have a distinct name, following Kubernetenes plugin naming scheme
    `<vendor name>/<provisioner name>`, e.g. `gluster.org/gluster-volume`.

  * The provisioner SHOULD send events on a claim to report any errors
    related to provisioning a volume for the claim. This way, users get the same
    experience as with internal provisioners.

  * The provisioner MUST implement also a deleter. It must be able to delete
    storage assets it created. It MUST NOT assume that any other internal or
    external plugin is present.

  The external provisioner runs in a separate process which watches claims, be
  it an external storage appliance, a daemon or a Kubernetes pod. For every
  claim creation or update, it implements these steps:

  1. The provisioner inspects if
     `claim.Annotations["volume.beta.kubernetes.io/storage-provisioner"] == <provisioner name>`.
     All other claims MUST be ignored.

  2. The provisioner MUST check that the claim is unbound, i.e. its
     `claim.Spec.VolumeName` is empty. Bound volumes MUST be ignored.

     *Race condition when the provisioner provisions a new PV for a claim and
     at the same time Kubernetes binds the same claim to another PV that was
     just created by admin is discussed below.*

  3. It tries to find a StorageClass instance referenced by annotation
     `claim.Annotations["volume.beta.kubernetes.io/storage-class"]`. If not
     found, it SHOULD report an error (by sending an event to the claim) and it
     SHOULD retry periodically with step i.

  4. The provisioner MUST parse arguments in the `StorageClass` and
     `claim.Spec.Selector` and provisions appropriate storage asset that matches
     both the parameters and the selector.
     When it encounters unknown parameters in `storageClass.Parameters` or
     `claim.Spec.Selector` or the combination of these parameters is impossible
     to achieve, it SHOULD report an error and it MUST NOT provision a volume.
     All errors found during parsing or provisioning SHOULD be send as events
     on the claim and the provisioner SHOULD retry periodically with step i.

     As parsing (and understanding) claim selectors is hard, the sentence
     "MUST parse ... `claim.Spec.Selector`"  will in typical case lead to simple
     refusal of claims that have any selector:

     ```go
     if pvc.Spec.Selector != nil {
        return Error("can't parse PVC selector!")
     }
     ```

  5. When the volume is provisioned, the provisioner MUST create a new PV
     representing  the storage asset and save it in Kubernetes. When this fails,
     it SHOULD retry creating the PV again few times. If all attempts fail, it
     MUST delete the storage asset. All errors SHOULD be sent as events to the
     claim.

     The created PV MUST have these properties:

     * `pv.Spec.ClaimRef` MUST point to the claim that led to its creation
       (including the claim UID).

       *This way, the PV will be bound to the claim.*

     * `pv.Annotations["pv.kubernetes.io/provisioned-by"]` MUST be set to name
       of the external provisioner. This provisioner will be used to delete the
       volume.

       *The provisioner/delete should not assume there is any other
       provisioner/deleter available that would delete the volume.*

     * `pv.Annotations["volume.beta.kubernetes.io/storage-class"]` MUST be set
       to name of the storage class requested by the claim.

       *So the created PV matches the claim.*

     * The provisioner MAY store any other information to the created PV as
       annotations. It SHOULD save any information that is needed to delete the
       storage asset there, as appropriate StorageClass instance may not exist
       when the volume will be deleted. However, references to Secret instance
       or direct username/password to a remote storage appliance MUST NOT be
       stored there, see issue #34822.

     * `pv.Labels` MUST be set to match `claim.spec.selector`. The provisioner
       MAY add additional labels.

       *So the created PV matches the claim.*

     * `pv.Spec` MUST be set to match requirements in `claim.Spec`, especially
       access mode and PV size. The provisioned volume size MUST NOT be smaller
       than size requested in the claim, however it MAY be larger.

       *So the created PV matches the claim.*

     * `pv.Spec.PersistentVolumeSource` MUST be set to point to the created
       storage asset.

     * `pv.Spec.PersistentVolumeReclaimPolicy` SHOULD be set to `Delete` unless
       user manually configures other reclaim policy.

     * `pv.Name` MUST be unique. Internal provisioners use name based on
       `claim.UID` to produce conflicts when two provisioners accidentally
       provision a PV for the same claim, however external provisioners can use
       any mechanism to generate an unique PV name.

  Example of a claim that is to be provisioned by an external provisioner for
  `foo.org/foo-volume`:

  ```yaml
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata:
    annotations:
      volume.beta.kubernetes.io/storage-class: myClass
      volume.beta.kubernetes.io/storage-provisioner: foo.org/foo-volume
    name: fooclaim
    namespace: default
    resourceVersion: "53"
    uid: 5a294561-7e5b-11e6-a20e-0eb6048532a3
  spec:
    accessModes:
    - ReadWriteOnce
    resources:
      requests:
        storage: 4Gi
  #  volumeName: must be empty!
  ```

  Example of the created PV:

  ```yaml
  apiVersion: v1
  kind: PersistentVolume
  metadata:
    annotations:
      pv.kubernetes.io/provisioned-by: foo.org/foo-volume
      volume.beta.kubernetes.io/storage-class: myClass
      foo.org/provisioner: "any other annotations as needed"
    labels:
        foo.org/my-label: "any labels as needed"
    generateName: "foo-volume-"
  spec:
    accessModes:
    - ReadWriteOnce
    awsElasticBlockStore:
      fsType: ext4
      volumeID: aws://us-east-1d/vol-de401a79
    capacity:
      storage: 4Gi
    claimRef:
      apiVersion: v1
      kind: PersistentVolumeClaim
      name: fooclaim
      namespace: default
      resourceVersion: "53"
      uid: 5a294561-7e5b-11e6-a20e-0eb6048532a3
    persistentVolumeReclaimPolicy: Delete
  ```

  As result, Kubernetes has a PV that represents the storage asset and is bound
  to the claim. When everything went well, Kubernetes completed binding of the
  claim to the PV.

  Kubernetes was not blocked in any way during the provisioning and could
  either bound the claim to another PV that was created by user or even the
  claim may have been deleted by the user. In both cases, Kubernetes will mark
  the PV to be delete using the protocol below.

  The external provisioner MAY save any annotations to the claim that is
  provisioned, however the claim may be modified or even deleted by the user at
  any time.


### Controller workflow for deleting volumes

When the controller decides that a volume should be deleted it performs these
steps:

1. The controller changes `pv.Status.Phase` to `Released`.

2. The controller looks for `pv.Annotations["pv.kubernetes.io/provisioned-by"]`.
   If found, it uses this provisioner/deleter to delete the volume.

3. If the volume is not annotated by `pv.kubernetes.io/provisioned-by`, the
   controller inspects `pv.Spec` and finds in-tree deleter for the volume.

4. If the deleter found by steps 2. or 3. is internal, it calls it and deletes
   the storage asset together with the PV that represents it.

5. If the deleter is not known to Kubernetes, it does not do anything.

6. External deleters MUST watch for PV changes. When
   `pv.Status.Phase == Released && pv.Annotations['pv.kubernetes.io/provisioned-by'] == <deleter name>`,
   the deleter:

   * It MUST check reclaim policy of the PV and ignore all PVs whose
     `Spec.PersistentVolumeReclaimPolicy` is not `Delete`.

   * It MUST delete the storage asset.

   * Only after the storage asset was successfully deleted, it MUST delete the
     PV object in Kubernetes.

   * Any error SHOULD be sent as an event on the PV being deleted and the
     deleter SHOULD retry to delete the volume periodically.

   * The deleter SHOULD NOT use any information from StorageClass instance
     referenced by the PV. This is different to internal deleters, which
     need to be StorageClass instance present at the time of deletion to read
     Secret instances (see Gluster provisioner for example), however we would
     like to phase out this behavior.

   Note that watching `pv.Status` has been frowned upon in the past, however in
   this particular case we could use it quite reliably to trigger deletion.
   It's not trivial to find out if a PV is not needed and should be deleted.
   *Alternatively, an annotation could be used.*

### Security considerations

Both internal and external provisioners and deleters may need access to
credentials (e.g. username+password) of an external storage appliance to
provision and delete volumes.

* For internal provisioners, a Secret instance in a well secured namespace
should be used. Pointer to the Secret instance shall be parameter of the
StorageClass and it MUST NOT be copied around the system e.g. in annotations
of PVs. See issue #34822.

* External provisioners running in pod should have appropriate credentials
mouted as Secret inside pods that run the provisioner. Namespace with the pods
and Secret instance should be well secured.

### `StorageClass` API

A new API group should hold the API for storage classes, following the pattern
of autoscaling, metrics, etc.  To allow for future storage-related APIs, we
should call this new API group `storage.k8s.io` and incubate in storage.k8s.io/v1beta1.

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

1. `claim.Spec.Selector` and `claim.Spec.Class` are mutually exclusive for now (1.4). User can either match existing volumes with `Selector` XOR match existing volumes with `Class` and get dynamic provisioning by using `Class`. This simplifies initial PR and also provisioners. This limitation may be lifted in future releases.

# Cloud Providers

Since the `volume.alpha.kubernetes.io/storage-class` is in use a `StorageClass` must be defined to support provisioning.  No default is assumed as before.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volume-provisioning.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
