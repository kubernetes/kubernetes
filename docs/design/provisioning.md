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

# Abstract

This document proposes a model for the configuration and management of dynamically provisioned persistent volumes in Kubernetes.  Familiarity [Persistent Volumes](../user-guide/persistent-volumes/) is assumed.

There are many kinds and classes of storage. Storage can be fast or slow, have different retention policies, be mounted in various ways, be manually created or dynamically provisioned, and more.  Each storage provider has distinct characteristics.  Administrators require the ability to define this diversity of storage with a flexible classification system that is easy to configure and easy to consume as an end user.

Use cases include:

* Allow configuration of many "storage classes" for both manually or dynamically created volumes
* Allow labelling and selection of volumes independent of storage classes
* Allow a single means of configuration and consumption of storage (i.e, make it easy for admins and end users)
* Allow easy update of storage config without server restart

Summary implementation:

Administrators use ConfigMap in a namespace they control to configure many provisioners.  Each provisioner creates 1 distinct kind and class of volume.  Using ConfigMap allows configuration to be stored in the API, which allows for easy update, watches, etc.

A PersistentVolumeSelector added to PersistentVolumeClaim would first attempt to match labels on available PVs before looking for a provisioner with a matching label set, allowing "storage classes" to be defined for both manually created and dynamically provisioned volumes.


## Using ConfigMap

### Admin namespace

Provisioners will be configured using `ConfigMap`.  `ConfigMap` allows configuration to be stored in the API server, which allows dynamic updates, querying, watches, etc.  The alternative is file config, which requires the same machinery as the API server (versioning, watches, querying, etc), or a new 1st class API object.  `ConfigMap` perfectly suits the requirements of this proposal without requiring a new API type.

`ConfigMap` objects live in a namespace, so the administrator will create a namespace specifically to hold provisioner configuration.  This "system namespace" may contain many types of ConfigMaps, not all of which are provisioners, so a specific label must be applied to provisioners to identify them.

The  namespace and label are required by kube-controller-manager so that the provisioner controller can query the correct namespace and find the provisioner configs.

Admins will pass the namespace and label to the controller via CLI flag:

```
CLI flags:

--pv-provisioner-namespace=foobar
--pv-provisioner-label="type=provisioner-config

example usage:

kubectl --namespace=system-namespace get configmap -l type=provisioner-config

```


### Provisioner Configuration

Use case:  Configure provisioners for GCE Persistent Disks in two zones (east and west) and two volume types (standard and solid state).  The admin chooses to use the term "storage-class" as a stand-in for volume type (speed) and uses the labels "gold" and "silver" for fast and slow.

A distinct set of labels on a provisioner is the "storage class".  The use case requires 4 provisioners: gold+east, gold+west, silver+east, silver+west.   Each provisioner's "parameters" map would contain the values for volume type and zone.  The plugin uses the parameters to create that one type of PersistentVolume.

One configuration is shown below.  The remainder can be found [here](provisioning-examples.json).

```
  {
    "kind": "ConfigMap",
    "apiVersion": "v1",
    "metadata": {
      "name": "gold-east",
      "namespace": "kubernetes.io/provisioners",
      "labels": {
        "type": "provisioner-config",
        "storage-class": "gold",
        "zone": "us-east"
      }
    },
    "data": {
      "plugin-name": "kubernetes.io/gce-pd",  
      "parameters": "{'volumeType':'ssd', 'zone':'us-east'}"  // OPAQUE JSON MAP OF PARAMS AS DEFINED BY THE PLUGIN
    }
  }
  
```

End users request specific storage by using a PersistentVolumeSelector on their claim.  This example shows a request for "gold" storage in the east zone.

```
{
  "kind": "PersistentVolumeClaim",
  "apiVersion": "v1",
  "metadata": {
    "name": "claim2",
    "namespace": "end-user-namespace"
  },
  "spec": {
    "accessModes": [
      "ReadWriteOnce"
    ],
    "persistentVolumeSelector":{
    	"matchLabels":{
    		"storage-class":"gold",
    		"zone":"us-east",
    	}
    }
    "resources": {
      "requests": {
        "storage": "3Gi"
      }
    }
  }
}
	
```

## Labelling PersistentVolumes

A `PersistentVolumeSelector` on claim has the additional benefit of being able to bind to volumes that are manually created as opposed to dynamically through a provisioner.

The same claim above (gold+east) can match on a PersistentVolume that is labelled similarly.

```
apiVersion: v1
  kind: PersistentVolume
  metadata:
    name: forMyClaim
    labels:
    	storage-class: gold
    	zone: us-east
  spec:
    capacity:
      storage: 15Gi
    accessModes:
      - ReadWriteOnce
    persistentVolumeReclaimPolicy: Recycle
    nfs:
      path: /tmp
      server: 172.17.0.2
```


## Controller behavior

`PersistentVolumeSelector` is being slightly overloaded in what it selects.  The binding controller would use the same selector on two different but related objects:  PersistentVolumes and Provisioners that create PersistentVolumes.

The binder attempts to fulfill a claim in the following order:

* If `pvc.Spec.PersistentVolumeSelector` is non-nil, attempt to find an available PV with a matching label set.
* If `pvc.Spec.PersistentVolumeSelector` is non-nil and matches a provisioner's labels, create a PV using the provisioner
* If `pvc.Spec.PersistentVolumeSelector` is nil, attempt to match an available PV by AccessModes and Capacity (current behavior).

Claims remain Pending indefinitely if a match is not found (with or without a selector).  The claim will be bound when a volume or provisioner is created that matches the claim.


## Design Considerations

`StorageClass` has been proposed as a field on PersistentVolumeClaim.  An equivalent of that field is currently stewing as an annotation in the current version of dynamic provisioning.

`PersistentVolumeSelector` on PersistentVolumeClaim has been proposed as a means to select specific volumes, akin to a pod's NodeSelector.

Having both fields is confusing.  Consider this on-premise installation of Kube:

> Admin Joe configures "Gold" to dynamically provision Ceph volumes.  Joe also creates "Silver" volumes by manually provisioning a large number of NFS exports on expensive hardware (SSDs, highly available, stringent backup policy).  Lastly, Joe creates "Bronze" NFS exports on older hardware with slower discs.

> How does a user request Gold/Silver/Bronze?   Using the StorageClass field for "Gold" while requiring the use of VolumeSelector for Silver/Bronze requires a user to know the difference between volumes.

Using just VolumeSelector gives a user a single path to understand.  Admins retain the ability to manually create storage.  Labels on the volume (or the volume creator) are matched with the claim's volume selector.


### Other approaches considered:

* Single ConfigMap with each key in Data representing 1 storage class, the value is opaque json.  Good for versioning the blob but it seemed unwieldy.
* Many ConfigMaps with "storage-class" as a key in the Data.  Difficult to validate unique storage names.
* Name of ConfigMap as name of storage class. Naming validation is enforced by API server.  Requires PVC.Spec.StorageClass (or similar) with 1:1 mapping of claim to provisioner.


## Plugins

Provisioning is implemented as plugins.  A number of standard plugins will be maintained within the Kubernetes codebase, but administrators may require implementations unique to their needs.  Future enhancements can include a DriverProvisioner to allow admins to create volumes that do not exist as plugins in the source tree.  The recent addition of FlexVolume is the prototype for this model.  That driver interface will improve over time and can eventually be made to provision volumes.

The following plugins will be provided by Kubernetes:

1.  EBSProvisioner -- creates a PV specifically for a claim and then creates the EBS volume in AWS.  Can be configured via parameters to create SSD, Provisioned IOPS, and Magnetic disks (EBS's 3 volume types).
2.  GCEProvisioner -- creates a PV specifically for a claim and then creates the GCEPersistentDisk in GCE.  Can be configured via parameters to create pd-standard and pd-ssd disks.
3.  CinderProvisioner -- creates a PV specifically for a claim and then creates a Cinder volume in OpenStack.  Can be configured via parameters to create multiple types (TODO: fill in exactly which types are supported)
4. CephProvisioner -- creates a PV specifically for a claim and then creates the CephVolume in the infrastructure.

### Plugin Parameters

The example above contains `"params": "${OPAQUE-JSON-MAP}"`, which is a string representing a JSON map of parameters.   Each provisionable volume plugin will document and publish the parameters it accepts.

Plugin params can vary by whatever attributes are exposed by a storage provider.  An AWS EBS volume, for example, can be a general purpose SSD (gp2), provisioned IOPs (io1) for speed and throughput, and magnetic (standard) for low cost.  Additionally, EBS volumes can optionally be encrypted.  Configuring each volume type with optional encryption would require 6 distinct provisioners.  Administrators can mix and match any attributes exposed by the provisioning plugin to create distinct storage classes.

# Tasks


1. Add pvc.Spec.PersistentVolumeSelector
2. Refactor controllers to use ConfigMaps
3. Refactor existing provisioning plugins, from 1 PV per plugin to Many using parameters.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/provisioning.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
