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
[here](http://releases.k8s.io/release-1.0/docs/design/persistent-volume-provisioning.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Persistent Volume Provisioning

This document proposes a model for dynamically provisioning

### tl;dr

One new API kind:

A `PersistentVolumeSet` (PVC) is a storage resource provisioned by an administrator.  PVCs do not have a namespace.  Just as a `ReplicationController` maintains a number of replicas of a pod, a `PersistentVolumeSet` maintains a minimum number of replicas of a `PersistentVolume`.  A PVC creates new volumes from a template up to a maximum replica count in increments of the minimum replica count.  A well-provisioned cluster will have many PVCs.

One new system component:

`PersistentVolumeSetManager` is a singleton control loop running in master that manages all PVControllers in the system.  The PVCM reconciles the current supply of available PersistentVolumes in the system with the desired levels according to the PVControllers.   This process is similar to the ```ReplicationManager``` that manages ReplicationControllers.

Three new volume plugin interfaces:

* Recycler -- knows how to scrub a volume clean so it can become available again as a resource
* Creator -- create new instances of a PV from a template.
* Deleter -- deletes instances of a PV and allows the plugin to determine how to remove it from the underlying infrastructure

Volume plugins can implement any applicable interfaces.  Each plugin will document its own support for dynamic provisioning.


### Goals

* Allow administrators to describe minimum and maximum storage levels comprised of many kinds of PersistentVolumes
* Allow the dynamic creation and reclamation of persistent volumes (to the fullest extent for each type of storage provider)


#### API types

```
// PersistentVolumeSet represents the configuration of a persistent volume.
type PersistentVolumeSet struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired specification of this PV controller.
	Spec PersistentVolumeSetSpec `json:"spec,omitempty"`

	// Status is the current status of this PV controller.
	Status PersistentVolumeSetStatus `json:"status,omitempty"`
}

// PersistentVolumeSetSpec is the specification of a PV controller.
// The controller creates volumes exactly like the Template until the unbound count equals the replica count
type PersistentVolumeSetSpec struct {
	// MinimumReplicas is the minimum number of unbound persistent volumes of this type desired in the system maintain
	MinimumReplicas int `json:"minimumReplicas"`

	// MaximumReplicas is the maximum total number of persistent volumes desired in the system
	MaximumReplicas int `json:"maximumReplicas"`
	
	// Selector is a label query over persistent volumes which are managed by this controller
	Selector map[string]string `json:"selector"`

	// Template is the description of a PersistentVolume to create new replicas from
	Template *PersistentVolumeTemplateSpec `json:"template,omitempty"`
}

// PersistentVolumeSetStatus represents the current status of a PVController
type PersistentVolumeSetStatus struct {
	// BoundReplicas is the number of replicas of this volume that are currently bound to PersistentVolumeClaims
	BoundReplicas int `json:"boundReplicas"`
	UnboundReplicas int `json:"unboundReplicas"`
}

// PersistentVolumeTemplateSpec describes the persistent volume created by this controller
type PersistentVolumeTemplateSpec struct {
	// Metadata of the PVs created from this template.
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines a persistent volume.
	Spec PersistentVolumeSpec `json:"spec,omitempty"`
}

```

Dynamic pools of storage (e.g, GCE) will maintain `MinimumReplicas` of a `PersistentVolume` at a given time where each PVs status phase is ` Available`, up to `MaximumReplicas`.  Storage levels will increase by one until the minimum or maximum levels are reached.  Recycling a PV means the current number of available volumes might exceed the minimum replica count.  Plugins implementing the Deleter interface will delete PVs where the `PersistentVolumeReclaimPolicy` is set to Delete.  The default reclaim policy is Retain.

Static pools of storage (e.g, NFS) will have ```MinimumReplicas``` == ```MaximumReplicas```.  Creation or deletion of the resource will not occur but the controller allows recycling of the volume through the plugin.

#### PersistentVolumeSet API

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/persistentvolumesets/ | Create instance of PersistentVolumeSet|
| GET | GET | /api/{version}persistentvolumesets/{name} | Get instance of PersistentVolumeSet |
| UPDATE | PUT | /api/{version}/persistentvolumesets/{name} | Update instance of PersistentVolumeSet |
| DELETE | DELETE | /api/{version}/persistentvolumesets/{name} | Delete instance of PersistentVolumeSet |
| LIST | GET | /api/{version}/persistentvolumesets | List instances of PersistentVolumeSet|
| WATCH | GET | /api/{version}/watch/persistentvolumesets | Watch for changes to a PersistentVolumeSet|


#### PersistentVolumeSetManager

* watch PersistentVolumes of a particular type as defined by the controller's label selector.
** PVs matching the selector but made manually count as replicas if they match the label selector
** PVs made manually or from earlier templates may have difference size capacities.  Delete largest first.
* create a new volume ``` if unboundReplicas < minimumReplicas && totalReplicas <= maximumReplicas```
* delete a volume ``` while totalReplicas > maximumReplicas, delete largest volume ```


#### Volumes and their plugins

Use of cloud and non-cloud volumes as examples for controllers.

| Volume |  Dynamic Creation | Formatting | Reclamation |
| ---- | ---- | ---- | ---- |
| GCE | yes via API | on mount, uses "safeFormatAndMount" | delete via API |
| AWS | yes via API | on mount, uses "safeFormatAndMount" | delete via API |
| OpenStack | yes via API | on mount, uses "safeFormatAndMount" | delete via API |
| NFS | No, externally provisioned | pre-formatted during provisioning | mount volume, run "rm -rf" on volume |
| ISCSI | No, externally provisioned | pre-formatted during provisioning | mount volume, run "rm -rf" on volume |
| gluster  | No, externally provisioned | pre-formatted during provisioning | mount volume, run "rm -rf" on volume | ? | ? | ? |
| ceph  | No, externally provisioned | pre-formatted during provisioning | mount volume, run "rm -rf" on volume | ? | ? | ? |

#### Recycler implementation

Recyclers will schedule and watch a pod that mounts the specific volume being recycled.  The scrubber pod runs "rm -rf" on the volume.
Recyclers block while performing their operation.  On successful return, the volume is made Available again and can be bound to a new `PersistentVolumeClaim`.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/persistent-volume-provisioning.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
