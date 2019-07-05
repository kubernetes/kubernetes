/*
Copyright 2016 The Kubernetes Authors.

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

package storage

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// StorageClass describes a named "class" of storage offered in a cluster.
// Different classes might map to quality-of-service levels, or to backup policies,
// or to arbitrary policies determined by the cluster administrators.  Kubernetes
// itself is unopinionated about what classes represent.  This concept is sometimes
// called "profiles" in other storage systems.
// The name of a StorageClass object is significant, and is how users can request a particular class.
type StorageClass struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// provisioner is the driver expected to handle this StorageClass.
	// This is an optionally-prefixed name, like a label key.
	// For example: "kubernetes.io/gce-pd" or "kubernetes.io/aws-ebs".
	// This value may not be empty.
	Provisioner string

	// parameters holds parameters for the provisioner.
	// These values are opaque to the  system and are passed directly
	// to the provisioner.  The only validation done on keys is that they are
	// not empty.  The maximum number of parameters is
	// 512, with a cumulative max size of 256K
	// +optional
	Parameters map[string]string

	// reclaimPolicy is the reclaim policy that dynamically provisioned
	// PersistentVolumes of this storage class are created with
	// +optional
	ReclaimPolicy *api.PersistentVolumeReclaimPolicy

	// mountOptions are the mount options that dynamically provisioned
	// PersistentVolumes of this storage class are created with
	// +optional
	MountOptions []string

	// AllowVolumeExpansion shows whether the storage class allow volume expand
	// If the field is nil or not set, it would amount to expansion disabled
	// for all PVs created from this storageclass.
	// +optional
	AllowVolumeExpansion *bool

	// VolumeBindingMode indicates how PersistentVolumeClaims should be
	// provisioned and bound.  When unset, VolumeBindingImmediate is used.
	// This field is only honored by servers that enable the VolumeScheduling feature.
	// +optional
	VolumeBindingMode *VolumeBindingMode

	// Restrict the node topologies where volumes can be dynamically provisioned.
	// Each volume plugin defines its own supported topology specifications.
	// An empty TopologySelectorTerm list means there is no topology restriction.
	// This field is only honored by servers that enable the VolumeScheduling feature.
	// +optional
	AllowedTopologies []api.TopologySelectorTerm
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// StorageClassList is a collection of storage classes.
type StorageClassList struct {
	metav1.TypeMeta
	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// Items is the list of StorageClasses
	Items []StorageClass
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Captures the intent to attach or detach the specified volume to/from
// the specified node.
//
// VolumeAttachment objects are non-namespaced.
type VolumeAttachment struct {
	metav1.TypeMeta

	// Standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta

	// Specification of the desired attach/detach volume behavior.
	// Populated by the Kubernetes system.
	Spec VolumeAttachmentSpec

	// Status of the VolumeAttachment request.
	// Populated by the entity completing the attach or detach
	// operation, i.e. the external-attacher.
	// +optional
	Status VolumeAttachmentStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// VolumeAttachmentList is a collection of VolumeAttachment objects.
type VolumeAttachmentList struct {
	metav1.TypeMeta
	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// Items is the list of VolumeAttachments
	Items []VolumeAttachment
}

// The specification of a VolumeAttachment request.
type VolumeAttachmentSpec struct {
	// Attacher indicates the name of the volume driver that MUST handle this
	// request. This is the name returned by GetPluginName().
	Attacher string

	// Source represents the volume that should be attached.
	Source VolumeAttachmentSource

	// The node that the volume should be attached to.
	NodeName string
}

// VolumeAttachmentSource represents a volume that should be attached.
// Right now persistent volumes as well as inline volumes (only in
// CSI Migration scenarios) can be attached via external attacher.
// Exactly one member can be set.
type VolumeAttachmentSource struct {
	// Name of the persistent volume to attach.
	// +optional
	PersistentVolumeName *string

	// inlineVolumeSpec contains all the information necessary to attach
	// a persistent volume defined by a pod's inline VolumeSource. This field
	// is populated only for the CSIMigration feature. It contains
	// translated fields from a pod's inline VolumeSource to a
	// PersistentVolumeSpec. This field is alpha-level and is only
	// honored by servers that enabled the CSIMigration feature.
	// +optional
	InlineVolumeSpec *api.PersistentVolumeSpec
}

// The status of a VolumeAttachment request.
type VolumeAttachmentStatus struct {
	// Indicates the volume is successfully attached.
	// This field must only be set by the entity completing the attach
	// operation, i.e. the external-attacher.
	Attached bool

	// Upon successful attach, this field is populated with any
	// information returned by the attach operation that must be passed
	// into subsequent WaitForAttach or Mount calls.
	// This field must only be set by the entity completing the attach
	// operation, i.e. the external-attacher.
	// +optional
	AttachmentMetadata map[string]string

	// The last error encountered during attach operation, if any.
	// This field must only be set by the entity completing the attach
	// operation, i.e. the external-attacher.
	// +optional
	AttachError *VolumeError

	// The last error encountered during detach operation, if any.
	// This field must only be set by the entity completing the detach
	// operation, i.e. the external-attacher.
	// +optional
	DetachError *VolumeError
}

// Captures an error encountered during a volume operation.
type VolumeError struct {
	// Time the error was encountered.
	// +optional
	Time metav1.Time

	// String detailing the error encountered during Attach or Detach operation.
	// This string may be logged, so it should not contain sensitive
	// information.
	// +optional
	Message string
}

// VolumeBindingMode indicates how PersistentVolumeClaims should be bound.
type VolumeBindingMode string

const (
	// VolumeBindingImmediate indicates that PersistentVolumeClaims should be
	// immediately provisioned and bound.
	VolumeBindingImmediate VolumeBindingMode = "Immediate"

	// VolumeBindingWaitForFirstConsumer indicates that PersistentVolumeClaims
	// should not be provisioned and bound until the first Pod is created that
	// references the PeristentVolumeClaim.  The volume provisioning and
	// binding will occur during Pod scheduing.
	VolumeBindingWaitForFirstConsumer VolumeBindingMode = "WaitForFirstConsumer"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CSIDriver captures information about a Container Storage Interface (CSI)
// volume driver deployed on the cluster.
// CSI drivers do not need to create the CSIDriver object directly. Instead they may use the
// cluster-driver-registrar sidecar container. When deployed with a CSI driver it automatically
// creates a CSIDriver object representing the driver.
// Kubernetes attach detach controller uses this object to determine whether attach is required.
// Kubelet uses this object to determine whether pod information needs to be passed on mount.
// CSIDriver objects are non-namespaced.
type CSIDriver struct {
	metav1.TypeMeta

	// Standard object metadata.
	// metadata.Name indicates the name of the CSI driver that this object
	// refers to; it MUST be the same name returned by the CSI GetPluginName()
	// call for that driver.
	// The driver name must be 63 characters or less, beginning and ending with
	// an alphanumeric character ([a-z0-9A-Z]) with dashes (-), dots (.), and
	// alphanumerics between.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	metav1.ObjectMeta

	// Specification of the CSI Driver.
	Spec CSIDriverSpec
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CSIDriverList is a collection of CSIDriver objects.
type CSIDriverList struct {
	metav1.TypeMeta

	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// items is the list of CSIDriver
	Items []CSIDriver
}

// CSIDriverSpec is the specification of a CSIDriver.
type CSIDriverSpec struct {
	// attachRequired indicates this CSI volume driver requires an attach
	// operation (because it implements the CSI ControllerPublishVolume()
	// method), and that the Kubernetes attach detach controller should call
	// the attach volume interface which checks the volumeattachment status
	// and waits until the volume is attached before proceeding to mounting.
	// The CSI external-attacher coordinates with CSI volume driver and updates
	// the volumeattachment status when the attach operation is complete.
	// If the CSIDriverRegistry feature gate is enabled and the value is
	// specified to false, the attach operation will be skipped.
	// Otherwise the attach operation will be called.
	// +optional
	AttachRequired *bool

	// If set to true, podInfoOnMount indicates this CSI volume driver
	// requires additional pod information (like podName, podUID, etc.) during
	// mount operations.
	// If set to false, pod information will not be passed on mount.
	// Default is false.
	// The CSI driver specifies podInfoOnMount as part of driver deployment.
	// If true, Kubelet will pass pod information as VolumeContext in the CSI
	// NodePublishVolume() calls.
	// The CSI driver is responsible for parsing and validating the information
	// passed in as VolumeContext.
	// The following VolumeConext will be passed if podInfoOnMount is set to true.
	// This list might grow, but the prefix will be used.
	// "csi.storage.k8s.io/pod.name": pod.Name
	// "csi.storage.k8s.io/pod.namespace": pod.Namespace
	// "csi.storage.k8s.io/pod.uid": string(pod.UID)
	// +optional
	PodInfoOnMount *bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CSINode holds information about all CSI drivers installed on a node.
// CSI drivers do not need to create the CSINode object directly. As long as
// they use the node-driver-registrar sidecar container, the kubelet will
// automatically populate the CSINode object for the CSI driver as part of
// kubelet plugin registration.
// CSINode has the same name as a node. If the object is missing, it means either
// there are no CSI Drivers available on the node, or the Kubelet version is low
// enough that it doesn't create this object.
// CSINode has an OwnerReference that points to the corresponding node object.
type CSINode struct {
	metav1.TypeMeta

	// metadata.name must be the Kubernetes node name.
	metav1.ObjectMeta

	// spec is the specification of CSINode
	Spec CSINodeSpec
}

// CSINodeSpec holds information about the specification of all CSI drivers installed on a node
type CSINodeSpec struct {
	// drivers is a list of information of all CSI Drivers existing on a node.
	// If all drivers in the list are uninstalled, this can become empty.
	// +patchMergeKey=name
	// +patchStrategy=merge
	Drivers []CSINodeDriver
}

// CSINodeDriver holds information about the specification of one CSI driver installed on a node
type CSINodeDriver struct {
	// This is the name of the CSI driver that this object refers to.
	// This MUST be the same name returned by the CSI GetPluginName() call for
	// that driver.
	Name string

	// nodeID of the node from the driver point of view.
	// This field enables Kubernetes to communicate with storage systems that do
	// not share the same nomenclature for nodes. For example, Kubernetes may
	// refer to a given node as "node1", but the storage system may refer to
	// the same node as "nodeA". When Kubernetes issues a command to the storage
	// system to attach a volume to a specific node, it can use this field to
	// refer to the node name using the ID that the storage system will
	// understand, e.g. "nodeA" instead of "node1". This field is required.
	NodeID string

	// topologyKeys is the list of keys supported by the driver.
	// When a driver is initialized on a cluster, it provides a set of topology
	// keys that it understands (e.g. "company.com/zone", "company.com/region").
	// When a driver is initialized on a node, it provides the same topology keys
	// along with values. Kubelet will expose these topology keys as labels
	// on its own node object.
	// When Kubernetes does topology aware provisioning, it can use this list to
	// determine which labels it should retrieve from the node object and pass
	// back to the driver.
	// It is possible for different nodes to use different topology keys.
	// This can be empty if driver does not support topology.
	// +optional
	TopologyKeys []string

	// allocatable represents the volume resources of a node that are available for scheduling.
	// +optional
	Allocatable *VolumeNodeResources
}

// VolumeNodeResources is a set of resource limits for scheduling of volumes.
type VolumeNodeResources struct {
	// Maximum number of unique volumes managed by the CSI driver that can be used on a node.
	// A volume that is both attached and mounted on a node is considered to be used once, not twice.
	// The same rule applies for a unique volume that is shared among multiple pods on the same node.
	// If this field is nil, then the supported number of volumes on this node is unbounded.
	// +optional
	Count *int32
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CSINodeList is a collection of CSINode objects.
type CSINodeList struct {
	metav1.TypeMeta

	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// items is the list of CSINode
	Items []CSINode
}
