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

Administrators use ConfigMap in a namespace they control to configure many provisioners.  Users request volumes via a selector and/or storage class.

## ConfigMap

Provisioners will be configured using `ConfigMap`.  Configs will be held in a namespace controlled by the administrator.

The admin configures kube-controller-manager with a CLI flag specifying the namespace containing the configs.

`--pv-provisioner-namespace=foobar`


### Example Configuration


```
	config := &api.ConfigMap {
		ObjectMeta: api.ObjectMeta{
			Name: "gold",
			Namespace: "foobar",
			Labels: {
				"storage-class": "gold",			}
		},
		Data: map[string]string {
			"plugin-name": "kubernetes.io/gce-pd",
			"service-account-name": "foo",
			"params": "${OPAQUE-JSON-MAP}",		}
	}
		

	claim := &api.PersistentVolumeClaim{
		Spec: api.PersistentVolumeClaimSpec{
				PersistentVolumeSelector:map[string]string{
					"storage-class":"gold",
				},
			},
		},
	}
	
```

#### Design Consideration

`StorageClass` has been proposed as a field on PersistentVolumeClaim.  An equivalent of that field is currently stewing as an annotation in the current version of dynamic provisioning.

`PersistentVolumeSelector` on PersistentVolumeClaim has been proposed as a means to select specific volumes, akin to a pod's NodeSelector.

Having both fields is confusing.  Consider this on-premise installation of Kube:

> Admin Joe configures "Gold" to dynamically provision Ceph volumes.  Joe also creates "Silver" volumes by manually provisioning a large number of NFS exports on expensive hardware (SSDs, highly available, stringent backup policy).  Lastly, Joe creates "Bronze" NFS exports on older hardware with slower discs.

> How does a user request Gold/Silver/Bronze?   Using the StorageClass field for "Gold" while requiring the use of VolumeSelector for Silver/Bronze requires a user to know the difference between volumes.

Using just VolumeSelector gives a user a single path to understand.  Admins retain the ability to manually create storage.  Labels on the volume (or the volume creator) are matched with the claim's volume selector.

An additional benefit of using a selector for everything is the ability to deprecate PersistentVolumeAccessModes.  These fields on PV are descriptors only.  They have no functional value other than to describe the capabilities of the volume.  AccessModes on a PVC represent only a way to select certain volumes.  Admins can use labels to describe a volume's capabilities (e.g, ReadWriteMany) and users would select volumes accordingly.





#### Other approaches considered:

* Single ConfigMap with each key in Data representing 1 storage class, the value is opaque json.  Good for versioning the blob but it seemed unwieldy.
* Many ConfigMaps with "storage-class" as a key in the Data.  Difficult to validate unique storage names.
* Name of ConfigMap as name of storage class. Naming validation is enforced by API server.  Requires PVC.Spec.StorageClass (or similar) with 1:1 mapping of claim to provisioner.



### Plugin Parameters

The example above contains `"params": "${OPAQUE-JSON-MAP}"`, which is a string representing a JSON map of parameters.   Each provisionable volume plugin will document and publish the parameters it accepts.

Plugin params can vary by whatever attributes are exposed by a storage provider.  An AWS EBS volume, for example, can be a general purpose SSD (gp2), provisioned IOPs (io1) for speed and throughput, and magnetic (standard) for low cost.  Additionally, EBS volumes can optionally be encrypted.  Configuring each volume type with optional encryption would require 6 distinct provisioners.  Administrators can mix and match any attributes exposed by the provisioning plugin to create distinct storage classes.


## Controller behavior

`PersistentVolumeBinderController` continues to watch claims and attempts to bind them to volumes, creating them as necessary/possible.

The binder attempts to fulfill a claim in the following order:

* If `pvc.Spec.PersistentVolumeSelector` is non-nil and matches a provisioner's labels, create a PV using the provisioner
* If `pvc.Spec.PersistentVolumeSelector` is non-nil but matches no provisioner, attempt to find an available PVs with a matching label set.
* If `pvc.Spec.PersistentVolumeSelector` is nil, attempt to match an available PV by AccessModes and Capacity (current behavior).


### Plugins

Provisioning is implemented as plugins.  A number of standard plugins will be maintained within the Kubernetes codebase, but administrators may require implementations unique to their needs.  Future enhancements can include a DriverProvisioner to allow admins to create volumes that do not exist as plugins in the source tree.  The recent addition of FlexVolume is the prototype for this model.  That driver interface will improve over time and can eventually be made to provision volumes.

The following plugins will be provided by Kubernetes:

1.  EBSProvisioner -- creates a PV specifically for a claim and then creates the EBS volume in AWS.  Can be configured via parameters to create SSD, Provisioned IOPS, and Magnetic disks (EBS's 3 volume types).
2.  GCEProvisioner -- creates a PV specifically for a claim and then creates the GCEPersistentDisk in GCE.  Can be configured via parameters to create pd-standard and pd-ssd disks.
3.  CinderProvisioner -- creates a PV specifically for a claim and then creates a Cinder volume in OpenStack.  Can be configured via parameters to create multiple types (TODO: fill in exactly which types are supported)
4. CephProvisioner -- creates a PV specifically for a claim and then creates the CephVolume in the infrastructure.


# Tasks


1. Add pvc.Spec.PersistentVolumeSelector
2. Refactor controllers to use ConfigMaps
3. Refactor existing provisioning plugins, from 1 PV per plugin to Many.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/provisioning.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
