/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeAttachment captures the intent to attach or detach the specified volume
// to/from the specified node.
//
// VolumeAttachment objects are non-namespaced.
type VolumeAttachment struct {
	metav1.TypeMeta `json:",inline"`

	// Standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired attach/detach volume behavior.
	// Populated by the Kubernetes system.
	Spec VolumeAttachmentSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status of the VolumeAttachment request.
	// Populated by the entity completing the attach or detach
	// operation, i.e. the external-attacher.
	// +optional
	Status VolumeAttachmentStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeAttachmentList is a collection of VolumeAttachment objects.
type VolumeAttachmentList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of VolumeAttachments
	Items []VolumeAttachment `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// VolumeAttachmentSpec is the specification of a VolumeAttachment request.
type VolumeAttachmentSpec struct {
	// Attacher indicates the name of the volume driver that MUST handle this
	// request. This is the name returned by GetPluginName().
	Attacher string `json:"attacher" protobuf:"bytes,1,opt,name=attacher"`

	// Source represents the volume that should be attached.
	Source VolumeAttachmentSource `json:"source" protobuf:"bytes,2,opt,name=source"`

	// The node that the volume should be attached to.
	NodeName string `json:"nodeName" protobuf:"bytes,3,opt,name=nodeName"`
}

// VolumeAttachmentSource represents a volume that should be attached.
// Right now only PersistenVolumes can be attached via external attacher,
// in future we may allow also inline volumes in pods.
// Exactly one member can be set.
type VolumeAttachmentSource struct {
	// Name of the persistent volume to attach.
	// +optional
	PersistentVolumeName *string `json:"persistentVolumeName,omitempty" protobuf:"bytes,1,opt,name=persistentVolumeName"`

	// Placeholder for *VolumeSource to accommodate inline volumes in pods.
}

// VolumeAttachmentStatus is the status of a VolumeAttachment request.
type VolumeAttachmentStatus struct {
	// Indicates the volume is successfully attached.
	// This field must only be set by the entity completing the attach
	// operation, i.e. the external-attacher.
	Attached bool `json:"attached" protobuf:"varint,1,opt,name=attached"`

	// Upon successful attach, this field is populated with any
	// information returned by the attach operation that must be passed
	// into subsequent WaitForAttach or Mount calls.
	// This field must only be set by the entity completing the attach
	// operation, i.e. the external-attacher.
	// +optional
	AttachmentMetadata map[string]string `json:"attachmentMetadata,omitempty" protobuf:"bytes,2,rep,name=attachmentMetadata"`

	// The last error encountered during attach operation, if any.
	// This field must only be set by the entity completing the attach
	// operation, i.e. the external-attacher.
	// +optional
	AttachError *VolumeError `json:"attachError,omitempty" protobuf:"bytes,3,opt,name=attachError,casttype=VolumeError"`

	// The last error encountered during detach operation, if any.
	// This field must only be set by the entity completing the detach
	// operation, i.e. the external-attacher.
	// +optional
	DetachError *VolumeError `json:"detachError,omitempty" protobuf:"bytes,4,opt,name=detachError,casttype=VolumeError"`
}

// VolumeError captures an error encountered during a volume operation.
type VolumeError struct {
	// Time the error was encountered.
	// +optional
	Time metav1.Time `json:"time,omitempty" protobuf:"bytes,1,opt,name=time"`

	// String detailing the error encountered during Attach or Detach operation.
	// This string maybe logged, so it should not contain sensitive
	// information.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,2,opt,name=message"`
}

// VolumeSnapshotStatus is the status of the VolumeSnapshot
type VolumeSnapshotStatus struct {
	// The time the snapshot was successfully created
	// +optional
	CreationTimestamp metav1.Time `json:"creationTimestamp" protobuf:"bytes,1,opt,name=creationTimestamp"`

	// Represent the latest available observations about the volume snapshot
	Conditions []VolumeSnapshotCondition `json:"conditions" protobuf:"bytes,2,rep,name=conditions"`
}

// VolumeSnapshotConditionType is the type of VolumeSnapshot conditions
type VolumeSnapshotConditionType string

// These are valid conditions of a volume snapshot.
const (
	// VolumeSnapshotConditionCreating means the snapshot is being created but
	// it is not cut yet.
	VolumeSnapshotConditionCreating VolumeSnapshotConditionType = "Creating"

	// VolumeSnapshotConditionUploading means the snapshot is cut and the application
	// can resume accessing data if ConditionStatus is True. It corresponds
	// to "Uploading" in GCE PD or "Pending" in AWS and ConditionStatus is True.
	// This condition type is not applicable in OpenStack Cinder.
	VolumeSnapshotConditionUploading VolumeSnapshotConditionType = "Uploading"
	// VolumeSnapshotConditionReady is added when the snapshot has been successfully created and is ready to be used.

	VolumeSnapshotConditionReady VolumeSnapshotConditionType = "Ready"
	// VolumeSnapshotConditionError means an error occurred during snapshot creation.

	VolumeSnapshotConditionError VolumeSnapshotConditionType = "Error"
)

// VolumeSnapshotCondition describes the state of a volume snapshot  at a certain point.
type VolumeSnapshotCondition struct {
	// Type of VolumeSnapshot condition.
	Type VolumeSnapshotConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=VolumeSnapshotConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=ConditionStatus"`
	// The last time the condition transitioned from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime" protobuf:"bytes,3,opt,name=lastTransitionTime"`
	// The reason for the condition's last transition.
	// +optional
	Reason string `json:"reason" protobuf:"bytes,4,opt,name=reason"`
	// A human readable message indicating details about the transition.
	// +optional
	Message string `json:"message" protobuf:"bytes,5,opt,name=message"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeSnapshot is the volume snapshot object accessible to the user. Upon succesful creation of the actual
// snapshot by the volume provider it is bound to the corresponding VolumeSnapshotData through
// the VolumeSnapshotSpec
type VolumeSnapshot struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec represents the desired state of the snapshot
	// +optional
	Spec VolumeSnapshotSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// Status represents the latest observer state of the snapshot
	// +optional
	Status VolumeSnapshotStatus `json:"status" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeSnapshotList is a list of VolumeSnapshot objects
type VolumeSnapshotList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []VolumeSnapshot `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// VolumeSnapshotSpec is the desired state of the volume snapshot
type VolumeSnapshotSpec struct {
	// PersistentVolumeClaimName is the name of the PVC being snapshotted
	// +optional
	PersistentVolumeClaimName string `json:"persistentVolumeClaimName" protobuf:"bytes,1,opt,name=persistentVolumeClaimName"`

	// SnapshotDataName binds the VolumeSnapshot object with the VolumeSnapshotData
	// +optional
	SnapshotDataName string `json:"snapshotDataName" protobuf:"bytes,2,opt,name=snapshotDataName"`

	// Name of the StorageClass required by the volume snapshot. This
	// StorageClass can be the same as or different from the one used in
	// the source persistent volume claim. If not specified, the StorageClass
	// in the persistent volume claim will be used for creating the snapshot.
	// +optional
	StorageClassName string `json:"storageClassName" protobuf:"bytes,3,opt,name=storageClassName"`
}

// VolumeSnapshotDataStatus is the actual state of the volume snapshot
type VolumeSnapshotDataStatus struct {
	// The time the snapshot was successfully created
	// +optional
	CreationTimestamp metav1.Time `json:"creationTimestamp" protobuf:"bytes,1,opt,name=creationTimestamp"`

	// Representes the lates available observations about the volume snapshot
	Conditions []VolumeSnapshotDataCondition `json:"conditions" protobuf:"bytes,2,rep,name=conditions"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeSnapshotDataList is a list of VolumeSnapshotData objects
type VolumeSnapshotDataList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []VolumeSnapshotData `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// VolumeSnapshotDataConditionType is the type of the VolumeSnapshotData condition
type VolumeSnapshotDataConditionType string

// These are valid conditions of a volume snapshot.
const (
	// VolumeSnapshotDataReady is added when the on-disk snapshot has been successfully created.
	VolumeSnapshotDataConditionReady VolumeSnapshotDataConditionType = "Ready"

	// VolumeSnapshotDataUploading is added when the on-disk snapshot has been successfully cut and is being uploaded.
	VolumeSnapshotDataConditionUploading VolumeSnapshotDataConditionType = "Uploading"

	// VolumeSnapshotDataError is added but the on-disk snapshot is failed to created
	VolumeSnapshotDataConditionError VolumeSnapshotDataConditionType = "Error"
)

// VolumeSnapshotDataCondition describes the state of a volume snapshot  at a certain point.
type VolumeSnapshotDataCondition struct {
	// Type of volume snapshot condition.
	Type VolumeSnapshotDataConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=VolumeSnapshotDataConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=ConditionStatus"`
	// The last time the condition transitioned from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime" protobuf:"bytes,3,opt,name=lastTransitionTime"`
	// The reason for the condition's last transition.
	// +optional
	Reason string `json:"reason" protobuf:"bytes,4,opt,name=reason"`
	// A human readable message indicating details about the transition.
	// +optional
	Message string `json:"message" protobuf:"bytes,5,opt,name=message"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeSnapshotData represents the actual "on-disk" snapshot object
type VolumeSnapshotData struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec represents the desired state of the snapshot
	// +optional
	Spec VolumeSnapshotDataSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Status represents the latest observed state of the snapshot
	// +optional
	Status VolumeSnapshotDataStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// VolumeSnapshotDataSpec is the spec of the volume snapshot data
type VolumeSnapshotDataSpec struct {
	// Source represents the location and type of the volume snapshot
	VolumeSnapshotDataSource `json:",inline" protobuf:"bytes,1,opt,name=volumeSnapshotDataSource"`

	// VolumeSnapshotRef is part of bi-directional binding between VolumeSnapshot
	// and VolumeSnapshotData
	// +optional
	VolumeSnapshotRef *v1.ObjectReference `json:"volumeSnapshotRef,omitempty" protobuf:"bytes,2,opt,name=volumeSnapshotRef"`

	// PersistentVolumeRef represents the PersistentVolume that the snapshot has been
	// taken from
	// +optional
	PersistentVolumeRef *v1.ObjectReference `json:"persistentVolumeRef,omitempty" protobuf:"bytes,3,opt,name=persistentVolumeRef"`
}

// HostPathVolumeSnapshotSource is HostPath volume snapshot source
type HostPathVolumeSnapshotSource struct {
	// Path represents a tar file that stores the HostPath volume source
	Path string `json:"snapshot,omitempty" protobuf:"bytes,1,opt,name=snapshot"`
}

// GlusterVolumeSnapshotSource is Gluster volume snapshot source
type GlusterVolumeSnapshotSource struct {
	// UniqueID represents a snapshot resource.
	SnapshotID string `json:"snapshotId,omitempty" protobuf:"bytes,1,opt,name=snapshotId"`
}

// AWSElasticBlockStoreVolumeSnapshotSource is AWS EBS volume snapshot source
type AWSElasticBlockStoreVolumeSnapshotSource struct {
	// Unique id of the persistent disk snapshot resource. Used to identify the disk snapshot in AWS
	SnapshotID string `json:"snapshotId,omitempty" protobuf:"bytes,1,opt,name=snapshotId"`
}

// CinderVolumeSnapshotSource is Cinder volume snapshot source
type CinderVolumeSnapshotSource struct {
	// Unique id of the cinder volume snapshot resource. Used to identify the snapshot in OpenStack
	SnapshotID string `json:"snapshotId,omitempty" protobuf:"bytes,1,opt,name=snapshotId"`
}

// GCEPersistentDiskSnapshotSource is GCE PD volume snapshot source
type GCEPersistentDiskSnapshotSource struct {
	// Unique id of the persistent disk snapshot resource. Used to identify the disk snapshot in GCE
	SnapshotName string `json:"snapshotId,omitempty" protobuf:"bytes,1,opt,name=snapshotId"`
}

// CSIVolumeSnapshotSource is CSI volume snapshot source
type CSIVolumeSnapshotSource struct {
	// Driver is the name of the driver to use for this snapshot.
	// Required.
	Driver string `json:"driver,omitempty" protobuf:"bytes,1,opt,name=driver"`

	// SnapshotHandle is the unique snapshot id returned by the CSI volume
	// plugin’s CreateSnapshot to refer to the snapshot on all subsequent calls.
	// Required.
	SnapshotHandle string `json:"snapshotHandle,omitempty" protobuf:"bytes,2,opt,name=snapshotHandle"`

	// CreatedAt is timestamp when the point-in-time snapshot is taken on the storage
	// system. The format of this field should be a Unix nanoseconds time
	// encoded as an int64. On Unix, the command `date +%s%N` returns
	// the  current time in nanoseconds since 1970-01-01 00:00:00 UTC.
	// This field is REQUIRED.
	CreatedAt int64 `json:"createdAt,omitempty" protobuf:"bytes,3,opt,name=createdAt"`
}

// VolumeSnapshotDataSource represents the actual location and type of the snapshot. Only one of its members may be specified.
type VolumeSnapshotDataSource struct {
	// HostPath represents a directory on the host.
	// Provisioned by a developer or tester.
	// This is useful for single-node development and testing only!
	// On-host storage is not supported in any way and WILL NOT WORK in a multi-node cluster.
	// More info: https://kubernetes.io/docs/concepts/storage/volumes#hostpath
	// +optional
	HostPath *HostPathVolumeSnapshotSource `json:"hostPath,omitempty" protobuf:"bytes,1,opt,name=hostPath"`
	//GlusterSnapshotSource represents a gluster snapshot resource
	// +optional
	GlusterSnapshotVolume *GlusterVolumeSnapshotSource `json:"glusterSnapshotVolume,omitempty" protobuf:"bytes,2,opt,name=glusterSnapshotVolume"`
	// AWSElasticBlockStore represents an AWS Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	// More info: https://kubernetes.io/docs/concepts/storage/volumes#awselasticblockstore
	// +optional
	AWSElasticBlockStore *AWSElasticBlockStoreVolumeSnapshotSource `json:"awsElasticBlockStore,omitempty" protobuf:"bytes,3,opt,name=awsElasticBlockStore"`
	// GCEPersistentDiskSnapshotSource represents an GCE PD snapshot resource
	// +optional
	GCEPersistentDiskSnapshot *GCEPersistentDiskSnapshotSource `json:"gcePersistentDisk,omitempty" protobuf:"bytes,4,opt,name=gcePersistentDisk"`
	// CinderVolumeSnapshotSource represents Cinder snapshot resource
	// +optional
	CinderSnapshot *CinderVolumeSnapshotSource `json:"cinderVolume,omitempty" protobuf:"bytes,5,opt,name=cinderVolume"`
	// CSISnapshot represents CSI snapshot resource
	// +optional
	CSISnapshot *CSIVolumeSnapshotSource `json:"csiSnapshot,omitempty" protobuf:"bytes,6,opt,name=csiSnapshot"`
}
