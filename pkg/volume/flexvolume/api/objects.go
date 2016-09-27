package api

import (
	"k8s.io/kubernetes/pkg/api"
)

// Probe plugin

type AttachmentPolicy string

const (
	// No attach/detach to do.
	AttachNone AttachmentPolicy = ""
	// Attach/detach calls issued from the controller.
	AttachRemote AttachmentPolicy = "Remote"
	// Attach/detach calls issued from the target node.
	AttachLocal AttachmentPolicy = "Local"
)

type FlexVolumeDriverCapabilities struct {
	AccessMode          api.PersistentVolumeAccessMode
	DynamicProvisioning bool
	Attachment          AttachmentPolicy
	CustomMount         bool // This driver has a custom mount/unmount logic. Mount/Unmount must be called.
	SELinux             bool
	OwnershipManagement bool
	Metrics             bool
}

type FlexVolumeDriverSpec struct {
	Name   string // name of the driver. Ex: ganesha-nfs
	Driver string // Actual driver path. Ex: ganesha/nfs
}

type FlexVolumeDriver struct {
	unversioned.TypeMeta
	v1.ObjectMeta
	Spec             FlexVolumeDriverSpec
	Capabilities     FlexVolumeDriverCapabilities
	SupportedOptions []string // Driver options supported.
}

// Provision/Create a volume

type FlexVolumeStatus string

type FlexVolume struct {
	unversioned.TypeMeta
	v1.ObjectMeta
	Spec   api.FlexVolumeSource
	Status FlexVolumeStatus
}

// Delete a volume

// Attach a volume

type FlexVolumeAttachment struct {
	unversioned.TypeMeta
	v1.ObjectMeta
	Host      string
	Device    string
	MountPath string
}

// Detach a volume

// Mount a volume

type FlexVolumeMount struct {
	MountPath string
}

// Unmount a volume

// Metrics call

type FlexVolumeMetrics struct {
	Capacity  *resource.Quantity
	Used      *resource.Quantity
	Available *resource.Quantity
}
