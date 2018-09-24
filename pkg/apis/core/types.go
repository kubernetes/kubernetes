/*
Copyright 2014 The Kubernetes Authors.

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

package core

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	// NamespaceDefault means the object is in the default namespace which is applied when not specified by clients
	NamespaceDefault string = "default"
	// NamespaceAll is the default argument to specify on a context when you want to list or filter resources across all namespaces
	NamespaceAll string = ""
	// NamespaceNone is the argument for a context when there is no namespace.
	NamespaceNone string = ""
	// NamespaceSystem is the system namespace where we place system components.
	NamespaceSystem string = "kube-system"
	// NamespacePublic is the namespace where we place public info (ConfigMaps)
	NamespacePublic string = "kube-public"
	// NamespaceNodeLease is the namespace where we place node lease objects (used for node heartbeats)
	NamespaceNodeLease string = "kube-node-lease"
	// TerminationMessagePathDefault means the default path to capture the application termination message running in a container
	TerminationMessagePathDefault string = "/dev/termination-log"
)

// Volume represents a named volume in a pod that may be accessed by any containers in the pod.
type Volume struct {
	// Required: This must be a DNS_LABEL.  Each volume in a pod must have
	// a unique name.
	Name string
	// The VolumeSource represents the location and type of a volume to mount.
	// This is optional for now. If not specified, the Volume is implied to be an EmptyDir.
	// This implied behavior is deprecated and will be removed in a future version.
	// +optional
	VolumeSource
}

// VolumeSource represents the source location of a volume to mount.
// Only one of its members may be specified.
type VolumeSource struct {
	// HostPath represents file or directory on the host machine that is
	// directly exposed to the container. This is generally used for system
	// agents or other privileged things that are allowed to see the host
	// machine. Most containers will NOT need this.
	// ---
	// TODO(jonesdl) We need to restrict who can use host directory mounts and who can/can not
	// mount host directories as read/write.
	// +optional
	HostPath *HostPathVolumeSource
	// EmptyDir represents a temporary directory that shares a pod's lifetime.
	// +optional
	EmptyDir *EmptyDirVolumeSource
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	// +optional
	GCEPersistentDisk *GCEPersistentDiskVolumeSource
	// AWSElasticBlockStore represents an AWS EBS disk that is attached to a
	// kubelet's host machine and then exposed to the pod.
	// +optional
	AWSElasticBlockStore *AWSElasticBlockStoreVolumeSource
	// GitRepo represents a git repository at a particular revision.
	// DEPRECATED: GitRepo is deprecated. To provision a container with a git repo, mount an
	// EmptyDir into an InitContainer that clones the repo using git, then mount the EmptyDir
	// into the Pod's container.
	// +optional
	GitRepo *GitRepoVolumeSource
	// Secret represents a secret that should populate this volume.
	// +optional
	Secret *SecretVolumeSource
	// NFS represents an NFS mount on the host that shares a pod's lifetime
	// +optional
	NFS *NFSVolumeSource
	// ISCSIVolumeSource represents an ISCSI Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	// +optional
	ISCSI *ISCSIVolumeSource
	// Glusterfs represents a Glusterfs mount on the host that shares a pod's lifetime
	// +optional
	Glusterfs *GlusterfsVolumeSource
	// PersistentVolumeClaimVolumeSource represents a reference to a PersistentVolumeClaim in the same namespace
	// +optional
	PersistentVolumeClaim *PersistentVolumeClaimVolumeSource
	// RBD represents a Rados Block Device mount on the host that shares a pod's lifetime
	// +optional
	RBD *RBDVolumeSource

	// Quobyte represents a Quobyte mount on the host that shares a pod's lifetime
	// +optional
	Quobyte *QuobyteVolumeSource

	// FlexVolume represents a generic volume resource that is
	// provisioned/attached using an exec based plugin.
	// +optional
	FlexVolume *FlexVolumeSource

	// Cinder represents a cinder volume attached and mounted on kubelets host machine
	// +optional
	Cinder *CinderVolumeSource

	// CephFS represents a Cephfs mount on the host that shares a pod's lifetime
	// +optional
	CephFS *CephFSVolumeSource

	// Flocker represents a Flocker volume attached to a kubelet's host machine. This depends on the Flocker control service being running
	// +optional
	Flocker *FlockerVolumeSource

	// DownwardAPI represents metadata about the pod that should populate this volume
	// +optional
	DownwardAPI *DownwardAPIVolumeSource
	// FC represents a Fibre Channel resource that is attached to a kubelet's host machine and then exposed to the pod.
	// +optional
	FC *FCVolumeSource
	// AzureFile represents an Azure File Service mount on the host and bind mount to the pod.
	// +optional
	AzureFile *AzureFileVolumeSource
	// ConfigMap represents a configMap that should populate this volume
	// +optional
	ConfigMap *ConfigMapVolumeSource
	// VsphereVolume represents a vSphere volume attached and mounted on kubelets host machine
	// +optional
	VsphereVolume *VsphereVirtualDiskVolumeSource
	// AzureDisk represents an Azure Data Disk mount on the host and bind mount to the pod.
	// +optional
	AzureDisk *AzureDiskVolumeSource
	// PhotonPersistentDisk represents a Photon Controller persistent disk attached and mounted on kubelets host machine
	PhotonPersistentDisk *PhotonPersistentDiskVolumeSource
	// Items for all in one resources secrets, configmaps, and downward API
	Projected *ProjectedVolumeSource
	// PortworxVolume represents a portworx volume attached and mounted on kubelets host machine
	// +optional
	PortworxVolume *PortworxVolumeSource
	// ScaleIO represents a ScaleIO persistent volume attached and mounted on Kubernetes nodes.
	// +optional
	ScaleIO *ScaleIOVolumeSource
	// StorageOS represents a StorageOS volume that is attached to the kubelet's host machine and mounted into the pod
	// +optional
	StorageOS *StorageOSVolumeSource
}

// Similar to VolumeSource but meant for the administrator who creates PVs.
// Exactly one of its members must be set.
type PersistentVolumeSource struct {
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	// +optional
	GCEPersistentDisk *GCEPersistentDiskVolumeSource
	// AWSElasticBlockStore represents an AWS EBS disk that is attached to a
	// kubelet's host machine and then exposed to the pod.
	// +optional
	AWSElasticBlockStore *AWSElasticBlockStoreVolumeSource
	// HostPath represents a directory on the host.
	// Provisioned by a developer or tester.
	// This is useful for single-node development and testing only!
	// On-host storage is not supported in any way and WILL NOT WORK in a multi-node cluster.
	// +optional
	HostPath *HostPathVolumeSource
	// Glusterfs represents a Glusterfs volume that is attached to a host and exposed to the pod
	// +optional
	Glusterfs *GlusterfsVolumeSource
	// NFS represents an NFS mount on the host that shares a pod's lifetime
	// +optional
	NFS *NFSVolumeSource
	// RBD represents a Rados Block Device mount on the host that shares a pod's lifetime
	// +optional
	RBD *RBDPersistentVolumeSource
	// Quobyte represents a Quobyte mount on the host that shares a pod's lifetime
	// +optional
	Quobyte *QuobyteVolumeSource
	// ISCSIPersistentVolumeSource represents an ISCSI resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	// +optional
	ISCSI *ISCSIPersistentVolumeSource
	// FlexVolume represents a generic volume resource that is
	// provisioned/attached using an exec based plugin.
	// +optional
	FlexVolume *FlexPersistentVolumeSource
	// Cinder represents a cinder volume attached and mounted on kubelets host machine
	// +optional
	Cinder *CinderPersistentVolumeSource
	// CephFS represents a Ceph FS mount on the host that shares a pod's lifetime
	// +optional
	CephFS *CephFSPersistentVolumeSource
	// FC represents a Fibre Channel resource that is attached to a kubelet's host machine and then exposed to the pod.
	// +optional
	FC *FCVolumeSource
	// Flocker represents a Flocker volume attached to a kubelet's host machine. This depends on the Flocker control service being running
	// +optional
	Flocker *FlockerVolumeSource
	// AzureFile represents an Azure File Service mount on the host and bind mount to the pod.
	// +optional
	AzureFile *AzureFilePersistentVolumeSource
	// VsphereVolume represents a vSphere volume attached and mounted on kubelets host machine
	// +optional
	VsphereVolume *VsphereVirtualDiskVolumeSource
	// AzureDisk represents an Azure Data Disk mount on the host and bind mount to the pod.
	// +optional
	AzureDisk *AzureDiskVolumeSource
	// PhotonPersistentDisk represents a Photon Controller persistent disk attached and mounted on kubelets host machine
	PhotonPersistentDisk *PhotonPersistentDiskVolumeSource
	// PortworxVolume represents a portworx volume attached and mounted on kubelets host machine
	// +optional
	PortworxVolume *PortworxVolumeSource
	// ScaleIO represents a ScaleIO persistent volume attached and mounted on Kubernetes nodes.
	// +optional
	ScaleIO *ScaleIOPersistentVolumeSource
	// Local represents directly-attached storage with node affinity
	// +optional
	Local *LocalVolumeSource
	// StorageOS represents a StorageOS volume that is attached to the kubelet's host machine and mounted into the pod
	// More info: https://releases.k8s.io/HEAD/examples/volumes/storageos/README.md
	// +optional
	StorageOS *StorageOSPersistentVolumeSource
	// CSI (Container Storage Interface) represents storage that handled by an external CSI driver (Beta feature).
	// +optional
	CSI *CSIPersistentVolumeSource
}

type PersistentVolumeClaimVolumeSource struct {
	// ClaimName is the name of a PersistentVolumeClaim in the same namespace as the pod using this volume
	ClaimName string
	// Optional: Defaults to false (read/write).  ReadOnly here
	// will force the ReadOnly setting in VolumeMounts
	// +optional
	ReadOnly bool
}

const (
	// BetaStorageClassAnnotation represents the beta/previous StorageClass annotation.
	// It's deprecated and will be removed in a future release. (#51440)
	BetaStorageClassAnnotation = "volume.beta.kubernetes.io/storage-class"

	// MountOptionAnnotation defines mount option annotation used in PVs
	MountOptionAnnotation = "volume.beta.kubernetes.io/mount-options"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type PersistentVolume struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	//Spec defines a persistent volume owned by the cluster
	// +optional
	Spec PersistentVolumeSpec

	// Status represents the current information about persistent volume.
	// +optional
	Status PersistentVolumeStatus
}

type PersistentVolumeSpec struct {
	// Resources represents the actual resources of the volume
	Capacity ResourceList
	// Source represents the location and type of a volume to mount.
	PersistentVolumeSource
	// AccessModes contains all ways the volume can be mounted
	// +optional
	AccessModes []PersistentVolumeAccessMode
	// ClaimRef is part of a bi-directional binding between PersistentVolume and PersistentVolumeClaim.
	// ClaimRef is expected to be non-nil when bound.
	// claim.VolumeName is the authoritative bind between PV and PVC.
	// When set to non-nil value, PVC.Spec.Selector of the referenced PVC is
	// ignored, i.e. labels of this PV do not need to match PVC selector.
	// +optional
	ClaimRef *ObjectReference
	// Optional: what happens to a persistent volume when released from its claim.
	// +optional
	PersistentVolumeReclaimPolicy PersistentVolumeReclaimPolicy
	// Name of StorageClass to which this persistent volume belongs. Empty value
	// means that this volume does not belong to any StorageClass.
	// +optional
	StorageClassName string
	// A list of mount options, e.g. ["ro", "soft"]. Not validated - mount will
	// simply fail if one is invalid.
	// +optional
	MountOptions []string
	// volumeMode defines if a volume is intended to be used with a formatted filesystem
	// or to remain in raw block state. Value of Filesystem is implied when not included in spec.
	// This is an alpha feature and may change in the future.
	// +optional
	VolumeMode *PersistentVolumeMode
	// NodeAffinity defines constraints that limit what nodes this volume can be accessed from.
	// This field influences the scheduling of pods that use this volume.
	// +optional
	NodeAffinity *VolumeNodeAffinity
}

// VolumeNodeAffinity defines constraints that limit what nodes this volume can be accessed from.
type VolumeNodeAffinity struct {
	// Required specifies hard node constraints that must be met.
	Required *NodeSelector
}

// PersistentVolumeReclaimPolicy describes a policy for end-of-life maintenance of persistent volumes
type PersistentVolumeReclaimPolicy string

const (
	// PersistentVolumeReclaimRecycle means the volume will be recycled back into the pool of unbound persistent volumes on release from its claim.
	// The volume plugin must support Recycling.
	// DEPRECATED: The PersistentVolumeReclaimRecycle called Recycle is being deprecated. See announcement here: https://groups.google.com/forum/#!topic/kubernetes-dev/uexugCza84I
	PersistentVolumeReclaimRecycle PersistentVolumeReclaimPolicy = "Recycle"
	// PersistentVolumeReclaimDelete means the volume will be deleted from Kubernetes on release from its claim.
	// The volume plugin must support Deletion.
	PersistentVolumeReclaimDelete PersistentVolumeReclaimPolicy = "Delete"
	// PersistentVolumeReclaimRetain means the volume will be left in its current phase (Released) for manual reclamation by the administrator.
	// The default policy is Retain.
	PersistentVolumeReclaimRetain PersistentVolumeReclaimPolicy = "Retain"
)

// PersistentVolumeMode describes how a volume is intended to be consumed, either Block or Filesystem.
type PersistentVolumeMode string

const (
	// PersistentVolumeBlock means the volume will not be formatted with a filesystem and will remain a raw block device.
	PersistentVolumeBlock PersistentVolumeMode = "Block"
	// PersistentVolumeFilesystem means the volume will be or is formatted with a filesystem.
	PersistentVolumeFilesystem PersistentVolumeMode = "Filesystem"
)

type PersistentVolumeStatus struct {
	// Phase indicates if a volume is available, bound to a claim, or released by a claim
	// +optional
	Phase PersistentVolumePhase
	// A human-readable message indicating details about why the volume is in this state.
	// +optional
	Message string
	// Reason is a brief CamelCase string that describes any failure and is meant for machine parsing and tidy display in the CLI
	// +optional
	Reason string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type PersistentVolumeList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta
	Items []PersistentVolume
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PersistentVolumeClaim is a user's request for and claim to a persistent volume
type PersistentVolumeClaim struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the volume requested by a pod author
	// +optional
	Spec PersistentVolumeClaimSpec

	// Status represents the current information about a claim
	// +optional
	Status PersistentVolumeClaimStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type PersistentVolumeClaimList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta
	Items []PersistentVolumeClaim
}

// PersistentVolumeClaimSpec describes the common attributes of storage devices
// and allows a Source for provider-specific attributes
type PersistentVolumeClaimSpec struct {
	// Contains the types of access modes required
	// +optional
	AccessModes []PersistentVolumeAccessMode
	// A label query over volumes to consider for binding. This selector is
	// ignored when VolumeName is set
	// +optional
	Selector *metav1.LabelSelector
	// Resources represents the minimum resources required
	// +optional
	Resources ResourceRequirements
	// VolumeName is the binding reference to the PersistentVolume backing this
	// claim. When set to non-empty value Selector is not evaluated
	// +optional
	VolumeName string
	// Name of the StorageClass required by the claim.
	// More info: https://kubernetes.io/docs/concepts/storage/persistent-volumes/#class-1
	// +optional
	StorageClassName *string
	// volumeMode defines what type of volume is required by the claim.
	// Value of Filesystem is implied when not included in claim spec.
	// This is an alpha feature and may change in the future.
	// +optional
	VolumeMode *PersistentVolumeMode
	// This field requires the VolumeSnapshotDataSource alpha feature gate to be
	// enabled and currently VolumeSnapshot is the only supported data source.
	// If the provisioner can support VolumeSnapshot data source, it will create
	// a new volume and data will be restored to the volume at the same time.
	// If the provisioner does not support VolumeSnapshot data source, volume will
	// not be created and the failure will be reported as an event.
	// In the future, we plan to support more data source types and the behavior
	// of the provisioner may change.
	// +optional
	DataSource *TypedLocalObjectReference
}

type PersistentVolumeClaimConditionType string

// These are valid conditions of Pvc
const (
	// An user trigger resize of pvc has been started
	PersistentVolumeClaimResizing PersistentVolumeClaimConditionType = "Resizing"
	// PersistentVolumeClaimFileSystemResizePending - controller resize is finished and a file system resize is pending on node
	PersistentVolumeClaimFileSystemResizePending PersistentVolumeClaimConditionType = "FileSystemResizePending"
)

type PersistentVolumeClaimCondition struct {
	Type   PersistentVolumeClaimConditionType
	Status ConditionStatus
	// +optional
	LastProbeTime metav1.Time
	// +optional
	LastTransitionTime metav1.Time
	// +optional
	Reason string
	// +optional
	Message string
}

type PersistentVolumeClaimStatus struct {
	// Phase represents the current phase of PersistentVolumeClaim
	// +optional
	Phase PersistentVolumeClaimPhase
	// AccessModes contains all ways the volume backing the PVC can be mounted
	// +optional
	AccessModes []PersistentVolumeAccessMode
	// Represents the actual resources of the underlying volume
	// +optional
	Capacity ResourceList
	// +optional
	Conditions []PersistentVolumeClaimCondition
}

type PersistentVolumeAccessMode string

const (
	// can be mounted read/write mode to exactly 1 host
	ReadWriteOnce PersistentVolumeAccessMode = "ReadWriteOnce"
	// can be mounted in read-only mode to many hosts
	ReadOnlyMany PersistentVolumeAccessMode = "ReadOnlyMany"
	// can be mounted in read/write mode to many hosts
	ReadWriteMany PersistentVolumeAccessMode = "ReadWriteMany"
)

type PersistentVolumePhase string

const (
	// used for PersistentVolumes that are not available
	VolumePending PersistentVolumePhase = "Pending"
	// used for PersistentVolumes that are not yet bound
	// Available volumes are held by the binder and matched to PersistentVolumeClaims
	VolumeAvailable PersistentVolumePhase = "Available"
	// used for PersistentVolumes that are bound
	VolumeBound PersistentVolumePhase = "Bound"
	// used for PersistentVolumes where the bound PersistentVolumeClaim was deleted
	// released volumes must be recycled before becoming available again
	// this phase is used by the persistent volume claim binder to signal to another process to reclaim the resource
	VolumeReleased PersistentVolumePhase = "Released"
	// used for PersistentVolumes that failed to be correctly recycled or deleted after being released from a claim
	VolumeFailed PersistentVolumePhase = "Failed"
)

type PersistentVolumeClaimPhase string

const (
	// used for PersistentVolumeClaims that are not yet bound
	ClaimPending PersistentVolumeClaimPhase = "Pending"
	// used for PersistentVolumeClaims that are bound
	ClaimBound PersistentVolumeClaimPhase = "Bound"
	// used for PersistentVolumeClaims that lost their underlying
	// PersistentVolume. The claim was bound to a PersistentVolume and this
	// volume does not exist any longer and all data on it was lost.
	ClaimLost PersistentVolumeClaimPhase = "Lost"
)

type HostPathType string

const (
	// For backwards compatible, leave it empty if unset
	HostPathUnset HostPathType = ""
	// If nothing exists at the given path, an empty directory will be created there
	// as needed with file mode 0755, having the same group and ownership with Kubelet.
	HostPathDirectoryOrCreate HostPathType = "DirectoryOrCreate"
	// A directory must exist at the given path
	HostPathDirectory HostPathType = "Directory"
	// If nothing exists at the given path, an empty file will be created there
	// as needed with file mode 0644, having the same group and ownership with Kubelet.
	HostPathFileOrCreate HostPathType = "FileOrCreate"
	// A file must exist at the given path
	HostPathFile HostPathType = "File"
	// A UNIX socket must exist at the given path
	HostPathSocket HostPathType = "Socket"
	// A character device must exist at the given path
	HostPathCharDev HostPathType = "CharDevice"
	// A block device must exist at the given path
	HostPathBlockDev HostPathType = "BlockDevice"
)

// Represents a host path mapped into a pod.
// Host path volumes do not support ownership management or SELinux relabeling.
type HostPathVolumeSource struct {
	// If the path is a symlink, it will follow the link to the real path.
	Path string
	// Defaults to ""
	Type *HostPathType
}

// Represents an empty directory for a pod.
// Empty directory volumes support ownership management and SELinux relabeling.
type EmptyDirVolumeSource struct {
	// TODO: Longer term we want to represent the selection of underlying
	// media more like a scheduling problem - user says what traits they
	// need, we give them a backing store that satisfies that.  For now
	// this will cover the most common needs.
	// Optional: what type of storage medium should back this directory.
	// The default is "" which means to use the node's default medium.
	// +optional
	Medium StorageMedium
	// Total amount of local storage required for this EmptyDir volume.
	// The size limit is also applicable for memory medium.
	// The maximum usage on memory medium EmptyDir would be the minimum value between
	// the SizeLimit specified here and the sum of memory limits of all containers in a pod.
	// The default is nil which means that the limit is undefined.
	// More info: http://kubernetes.io/docs/user-guide/volumes#emptydir
	// +optional
	SizeLimit *resource.Quantity
}

// StorageMedium defines ways that storage can be allocated to a volume.
type StorageMedium string

const (
	StorageMediumDefault   StorageMedium = ""          // use whatever the default is for the node
	StorageMediumMemory    StorageMedium = "Memory"    // use memory (tmpfs)
	StorageMediumHugePages StorageMedium = "HugePages" // use hugepages
)

// Protocol defines network protocols supported for things like container ports.
type Protocol string

const (
	// ProtocolTCP is the TCP protocol.
	ProtocolTCP Protocol = "TCP"
	// ProtocolUDP is the UDP protocol.
	ProtocolUDP Protocol = "UDP"
	// ProtocolSCTP is the SCTP protocol.
	ProtocolSCTP Protocol = "SCTP"
)

// Represents a Persistent Disk resource in Google Compute Engine.
//
// A GCE PD must exist before mounting to a container. The disk must
// also be in the same GCE project and zone as the kubelet. A GCE PD
// can only be mounted as read/write once or read-only many times. GCE
// PDs support ownership management and SELinux relabeling.
type GCEPersistentDiskVolumeSource struct {
	// Unique name of the PD resource. Used to identify the disk in GCE
	PDName string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// +optional
	FSType string
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field is 0 or empty.
	// +optional
	Partition int32
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// Represents an ISCSI disk.
// ISCSI volumes can only be mounted as read/write once.
// ISCSI volumes support ownership management and SELinux relabeling.
type ISCSIVolumeSource struct {
	// Required: iSCSI target portal
	// the portal is either an IP or ip_addr:port if port is other than default (typically TCP ports 860 and 3260)
	// +optional
	TargetPortal string
	// Required:  target iSCSI Qualified Name
	// +optional
	IQN string
	// Required: iSCSI target lun number
	// +optional
	Lun int32
	// Optional: Defaults to 'default' (tcp). iSCSI interface name that uses an iSCSI transport.
	// +optional
	ISCSIInterface string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// +optional
	FSType string
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// Optional: list of iSCSI target portal ips for high availability.
	// the portal is either an IP or ip_addr:port if port is other than default (typically TCP ports 860 and 3260)
	// +optional
	Portals []string
	// Optional: whether support iSCSI Discovery CHAP authentication
	// +optional
	DiscoveryCHAPAuth bool
	// Optional: whether support iSCSI Session CHAP authentication
	// +optional
	SessionCHAPAuth bool
	// Optional: CHAP secret for iSCSI target and initiator authentication.
	// The secret is used if either DiscoveryCHAPAuth or SessionCHAPAuth is true
	// +optional
	SecretRef *LocalObjectReference
	// Optional: Custom initiator name per volume.
	// If initiatorName is specified with iscsiInterface simultaneously, new iSCSI interface
	// <target portal>:<volume name> will be created for the connection.
	// +optional
	InitiatorName *string
}

// ISCSIPersistentVolumeSource represents an ISCSI disk.
// ISCSI volumes can only be mounted as read/write once.
// ISCSI volumes support ownership management and SELinux relabeling.
type ISCSIPersistentVolumeSource struct {
	// Required: iSCSI target portal
	// the portal is either an IP or ip_addr:port if port is other than default (typically TCP ports 860 and 3260)
	// +optional
	TargetPortal string
	// Required:  target iSCSI Qualified Name
	// +optional
	IQN string
	// Required: iSCSI target lun number
	// +optional
	Lun int32
	// Optional: Defaults to 'default' (tcp). iSCSI interface name that uses an iSCSI transport.
	// +optional
	ISCSIInterface string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// +optional
	FSType string
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// Optional: list of iSCSI target portal ips for high availability.
	// the portal is either an IP or ip_addr:port if port is other than default (typically TCP ports 860 and 3260)
	// +optional
	Portals []string
	// Optional: whether support iSCSI Discovery CHAP authentication
	// +optional
	DiscoveryCHAPAuth bool
	// Optional: whether support iSCSI Session CHAP authentication
	// +optional
	SessionCHAPAuth bool
	// Optional: CHAP secret for iSCSI target and initiator authentication.
	// The secret is used if either DiscoveryCHAPAuth or SessionCHAPAuth is true
	// +optional
	SecretRef *SecretReference
	// Optional: Custom initiator name per volume.
	// If initiatorName is specified with iscsiInterface simultaneously, new iSCSI interface
	// <target portal>:<volume name> will be created for the connection.
	// +optional
	InitiatorName *string
}

// Represents a Fibre Channel volume.
// Fibre Channel volumes can only be mounted as read/write once.
// Fibre Channel volumes support ownership management and SELinux relabeling.
type FCVolumeSource struct {
	// Optional: FC target worldwide names (WWNs)
	// +optional
	TargetWWNs []string
	// Optional: FC target lun number
	// +optional
	Lun *int32
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// +optional
	FSType string
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// Optional: FC volume World Wide Identifiers (WWIDs)
	// Either WWIDs or TargetWWNs and Lun must be set, but not both simultaneously.
	// +optional
	WWIDs []string
}

// FlexPersistentVolumeSource represents a generic persistent volume resource that is
// provisioned/attached using an exec based plugin.
type FlexPersistentVolumeSource struct {
	// Driver is the name of the driver to use for this volume.
	Driver string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". The default filesystem depends on FlexVolume script.
	// +optional
	FSType string
	// Optional: SecretRef is reference to the secret object containing
	// sensitive information to pass to the plugin scripts. This may be
	// empty if no secret object is specified. If the secret object
	// contains more than one secret, all secrets are passed to the plugin
	// scripts.
	// +optional
	SecretRef *SecretReference
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// Optional: Extra driver options if any.
	// +optional
	Options map[string]string
}

// FlexVolume represents a generic volume resource that is
// provisioned/attached using an exec based plugin.
type FlexVolumeSource struct {
	// Driver is the name of the driver to use for this volume.
	Driver string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". The default filesystem depends on FlexVolume script.
	// +optional
	FSType string
	// Optional: SecretRef is reference to the secret object containing
	// sensitive information to pass to the plugin scripts. This may be
	// empty if no secret object is specified. If the secret object
	// contains more than one secret, all secrets are passed to the plugin
	// scripts.
	// +optional
	SecretRef *LocalObjectReference
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// Optional: Extra driver options if any.
	// +optional
	Options map[string]string
}

// Represents a Persistent Disk resource in AWS.
//
// An AWS EBS disk must exist before mounting to a container. The disk
// must also be in the same AWS zone as the kubelet. An AWS EBS disk
// can only be mounted as read/write once. AWS EBS volumes support
// ownership management and SELinux relabeling.
type AWSElasticBlockStoreVolumeSource struct {
	// Unique id of the persistent disk resource. Used to identify the disk in AWS
	VolumeID string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// +optional
	FSType string
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field is 0 or empty.
	// +optional
	Partition int32
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// Represents a volume that is populated with the contents of a git repository.
// Git repo volumes do not support ownership management.
// Git repo volumes support SELinux relabeling.
//
// DEPRECATED: GitRepo is deprecated. To provision a container with a git repo, mount an
// EmptyDir into an InitContainer that clones the repo using git, then mount the EmptyDir
// into the Pod's container.
type GitRepoVolumeSource struct {
	// Repository URL
	Repository string
	// Commit hash, this is optional
	// +optional
	Revision string
	// Clone target, this is optional
	// Must not contain or start with '..'.  If '.' is supplied, the volume directory will be the
	// git repository.  Otherwise, if specified, the volume will contain the git repository in
	// the subdirectory with the given name.
	// +optional
	Directory string
	// TODO: Consider credentials here.
}

// Adapts a Secret into a volume.
//
// The contents of the target Secret's Data field will be presented in a volume
// as files using the keys in the Data field as the file names.
// Secret volumes support ownership management and SELinux relabeling.
type SecretVolumeSource struct {
	// Name of the secret in the pod's namespace to use.
	// +optional
	SecretName string
	// If unspecified, each key-value pair in the Data field of the referenced
	// Secret will be projected into the volume as a file whose name is the
	// key and content is the value. If specified, the listed keys will be
	// projected into the specified paths, and unlisted keys will not be
	// present. If a key is specified which is not present in the Secret,
	// the volume setup will error unless it is marked optional. Paths must be
	// relative and may not contain the '..' path or start with '..'.
	// +optional
	Items []KeyToPath
	// Mode bits to use on created files by default. Must be a value between
	// 0 and 0777.
	// Directories within the path are not affected by this setting.
	// This might be in conflict with other options that affect the file
	// mode, like fsGroup, and the result can be other mode bits set.
	// +optional
	DefaultMode *int32
	// Specify whether the Secret or its key must be defined
	// +optional
	Optional *bool
}

// Adapts a secret into a projected volume.
//
// The contents of the target Secret's Data field will be presented in a
// projected volume as files using the keys in the Data field as the file names.
// Note that this is identical to a secret volume source without the default
// mode.
type SecretProjection struct {
	LocalObjectReference
	// If unspecified, each key-value pair in the Data field of the referenced
	// Secret will be projected into the volume as a file whose name is the
	// key and content is the value. If specified, the listed keys will be
	// projected into the specified paths, and unlisted keys will not be
	// present. If a key is specified which is not present in the Secret,
	// the volume setup will error unless it is marked optional. Paths must be
	// relative and may not contain the '..' path or start with '..'.
	// +optional
	Items []KeyToPath
	// Specify whether the Secret or its key must be defined
	// +optional
	Optional *bool
}

// Represents an NFS mount that lasts the lifetime of a pod.
// NFS volumes do not support ownership management or SELinux relabeling.
type NFSVolumeSource struct {
	// Server is the hostname or IP address of the NFS server
	Server string

	// Path is the exported NFS share
	Path string

	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the NFS export to be mounted with read-only permissions
	// +optional
	ReadOnly bool
}

// Represents a Quobyte mount that lasts the lifetime of a pod.
// Quobyte volumes do not support ownership management or SELinux relabeling.
type QuobyteVolumeSource struct {
	// Registry represents a single or multiple Quobyte Registry services
	// specified as a string as host:port pair (multiple entries are separated with commas)
	// which acts as the central registry for volumes
	Registry string

	// Volume is a string that references an already created Quobyte volume by name.
	Volume string

	// Defaults to false (read/write). ReadOnly here will force
	// the Quobyte to be mounted with read-only permissions
	// +optional
	ReadOnly bool

	// User to map volume access to
	// Defaults to the root user
	// +optional
	User string

	// Group to map volume access to
	// Default is no group
	// +optional
	Group string
}

// Represents a Glusterfs mount that lasts the lifetime of a pod.
// Glusterfs volumes do not support ownership management or SELinux relabeling.
type GlusterfsVolumeSource struct {
	// Required: EndpointsName is the endpoint name that details Glusterfs topology
	EndpointsName string

	// Required: Path is the Glusterfs volume path
	Path string

	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the Glusterfs to be mounted with read-only permissions
	// +optional
	ReadOnly bool
}

// Represents a Rados Block Device mount that lasts the lifetime of a pod.
// RBD volumes support ownership management and SELinux relabeling.
type RBDVolumeSource struct {
	// Required: CephMonitors is a collection of Ceph monitors
	CephMonitors []string
	// Required: RBDImage is the rados image name
	RBDImage string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// +optional
	FSType string
	// Optional: RadosPool is the rados pool name,default is rbd
	// +optional
	RBDPool string
	// Optional: RBDUser is the rados user name, default is admin
	// +optional
	RadosUser string
	// Optional: Keyring is the path to key ring for RBDUser, default is /etc/ceph/keyring
	// +optional
	Keyring string
	// Optional: SecretRef is name of the authentication secret for RBDUser, default is nil.
	// +optional
	SecretRef *LocalObjectReference
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// Represents a Rados Block Device mount that lasts the lifetime of a pod.
// RBD volumes support ownership management and SELinux relabeling.
type RBDPersistentVolumeSource struct {
	// Required: CephMonitors is a collection of Ceph monitors
	CephMonitors []string
	// Required: RBDImage is the rados image name
	RBDImage string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// +optional
	FSType string
	// Optional: RadosPool is the rados pool name,default is rbd
	// +optional
	RBDPool string
	// Optional: RBDUser is the rados user name, default is admin
	// +optional
	RadosUser string
	// Optional: Keyring is the path to key ring for RBDUser, default is /etc/ceph/keyring
	// +optional
	Keyring string
	// Optional: SecretRef is reference to the authentication secret for User, default is empty.
	// +optional
	SecretRef *SecretReference
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// Represents a cinder volume resource in Openstack. A Cinder volume
// must exist before mounting to a container. The volume must also be
// in the same region as the kubelet. Cinder volumes support ownership
// management and SELinux relabeling.
type CinderVolumeSource struct {
	// Unique id of the volume used to identify the cinder volume
	VolumeID string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// +optional
	FSType string
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// Optional: points to a secret object containing parameters used to connect
	// to OpenStack.
	// +optional
	SecretRef *LocalObjectReference
}

// Represents a cinder volume resource in Openstack. A Cinder volume
// must exist before mounting to a container. The volume must also be
// in the same region as the kubelet. Cinder volumes support ownership
// management and SELinux relabeling.
type CinderPersistentVolumeSource struct {
	// Unique id of the volume used to identify the cinder volume
	VolumeID string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// +optional
	FSType string
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// Optional: points to a secret object containing parameters used to connect
	// to OpenStack.
	// +optional
	SecretRef *SecretReference
}

// Represents a Ceph Filesystem mount that lasts the lifetime of a pod
// Cephfs volumes do not support ownership management or SELinux relabeling.
type CephFSVolumeSource struct {
	// Required: Monitors is a collection of Ceph monitors
	Monitors []string
	// Optional: Used as the mounted root, rather than the full Ceph tree, default is /
	// +optional
	Path string
	// Optional: User is the rados user name, default is admin
	// +optional
	User string
	// Optional: SecretFile is the path to key ring for User, default is /etc/ceph/user.secret
	// +optional
	SecretFile string
	// Optional: SecretRef is reference to the authentication secret for User, default is empty.
	// +optional
	SecretRef *LocalObjectReference
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// SecretReference represents a Secret Reference. It has enough information to retrieve secret
// in any namespace
type SecretReference struct {
	// Name is unique within a namespace to reference a secret resource.
	// +optional
	Name string
	// Namespace defines the space within which the secret name must be unique.
	// +optional
	Namespace string
}

// Represents a Ceph Filesystem mount that lasts the lifetime of a pod
// Cephfs volumes do not support ownership management or SELinux relabeling.
type CephFSPersistentVolumeSource struct {
	// Required: Monitors is a collection of Ceph monitors
	Monitors []string
	// Optional: Used as the mounted root, rather than the full Ceph tree, default is /
	// +optional
	Path string
	// Optional: User is the rados user name, default is admin
	// +optional
	User string
	// Optional: SecretFile is the path to key ring for User, default is /etc/ceph/user.secret
	// +optional
	SecretFile string
	// Optional: SecretRef is reference to the authentication secret for User, default is empty.
	// +optional
	SecretRef *SecretReference
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// Represents a Flocker volume mounted by the Flocker agent.
// One and only one of datasetName and datasetUUID should be set.
// Flocker volumes do not support ownership management or SELinux relabeling.
type FlockerVolumeSource struct {
	// Name of the dataset stored as metadata -> name on the dataset for Flocker
	// should be considered as deprecated
	// +optional
	DatasetName string
	// UUID of the dataset. This is unique identifier of a Flocker dataset
	// +optional
	DatasetUUID string
}

// Represents a volume containing downward API info.
// Downward API volumes support ownership management and SELinux relabeling.
type DownwardAPIVolumeSource struct {
	// Items is a list of DownwardAPIVolume file
	// +optional
	Items []DownwardAPIVolumeFile
	// Mode bits to use on created files by default. Must be a value between
	// 0 and 0777.
	// Directories within the path are not affected by this setting.
	// This might be in conflict with other options that affect the file
	// mode, like fsGroup, and the result can be other mode bits set.
	// +optional
	DefaultMode *int32
}

// Represents a single file containing information from the downward API
type DownwardAPIVolumeFile struct {
	// Required: Path is  the relative path name of the file to be created. Must not be absolute or contain the '..' path. Must be utf-8 encoded. The first item of the relative path must not start with '..'
	Path string
	// Required: Selects a field of the pod: only annotations, labels, name, namespace and uid are supported.
	// +optional
	FieldRef *ObjectFieldSelector
	// Selects a resource of the container: only resources limits and requests
	// (limits.cpu, limits.memory, requests.cpu and requests.memory) are currently supported.
	// +optional
	ResourceFieldRef *ResourceFieldSelector
	// Optional: mode bits to use on this file, must be a value between 0
	// and 0777. If not specified, the volume defaultMode will be used.
	// This might be in conflict with other options that affect the file
	// mode, like fsGroup, and the result can be other mode bits set.
	// +optional
	Mode *int32
}

// Represents downward API info for projecting into a projected volume.
// Note that this is identical to a downwardAPI volume source without the default
// mode.
type DownwardAPIProjection struct {
	// Items is a list of DownwardAPIVolume file
	// +optional
	Items []DownwardAPIVolumeFile
}

// AzureFile represents an Azure File Service mount on the host and bind mount to the pod.
type AzureFileVolumeSource struct {
	// the name of secret that contains Azure Storage Account Name and Key
	SecretName string
	// Share Name
	ShareName string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// AzureFile represents an Azure File Service mount on the host and bind mount to the pod.
type AzureFilePersistentVolumeSource struct {
	// the name of secret that contains Azure Storage Account Name and Key
	SecretName string
	// Share Name
	ShareName string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// the namespace of the secret that contains Azure Storage Account Name and Key
	// default is the same as the Pod
	// +optional
	SecretNamespace *string
}

// Represents a vSphere volume resource.
type VsphereVirtualDiskVolumeSource struct {
	// Path that identifies vSphere volume vmdk
	VolumePath string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// +optional
	FSType string
	// Storage Policy Based Management (SPBM) profile name.
	// +optional
	StoragePolicyName string
	// Storage Policy Based Management (SPBM) profile ID associated with the StoragePolicyName.
	// +optional
	StoragePolicyID string
}

// Represents a Photon Controller persistent disk resource.
type PhotonPersistentDiskVolumeSource struct {
	// ID that identifies Photon Controller persistent disk
	PdID string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	FSType string
}

// PortworxVolumeSource represents a Portworx volume resource.
type PortworxVolumeSource struct {
	// VolumeID uniquely identifies a Portworx volume
	VolumeID string
	// FSType represents the filesystem type to mount
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs". Implicitly inferred to be "ext4" if unspecified.
	// +optional
	FSType string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

type AzureDataDiskCachingMode string
type AzureDataDiskKind string

const (
	AzureDataDiskCachingNone      AzureDataDiskCachingMode = "None"
	AzureDataDiskCachingReadOnly  AzureDataDiskCachingMode = "ReadOnly"
	AzureDataDiskCachingReadWrite AzureDataDiskCachingMode = "ReadWrite"

	AzureSharedBlobDisk    AzureDataDiskKind = "Shared"
	AzureDedicatedBlobDisk AzureDataDiskKind = "Dedicated"
	AzureManagedDisk       AzureDataDiskKind = "Managed"
)

// AzureDisk represents an Azure Data Disk mount on the host and bind mount to the pod.
type AzureDiskVolumeSource struct {
	// The Name of the data disk in the blob storage
	DiskName string
	// The URI of the data disk in the blob storage
	DataDiskURI string
	// Host Caching mode: None, Read Only, Read Write.
	// +optional
	CachingMode *AzureDataDiskCachingMode
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// +optional
	FSType *string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly *bool
	// Expected values Shared: multiple blob disks per storage account  Dedicated: single blob disk per storage account  Managed: azure managed data disk (only in managed availability set). defaults to shared
	Kind *AzureDataDiskKind
}

// ScaleIOVolumeSource represents a persistent ScaleIO volume
type ScaleIOVolumeSource struct {
	// The host address of the ScaleIO API Gateway.
	Gateway string
	// The name of the storage system as configured in ScaleIO.
	System string
	// SecretRef references to the secret for ScaleIO user and other
	// sensitive information. If this is not provided, Login operation will fail.
	SecretRef *LocalObjectReference
	// Flag to enable/disable SSL communication with Gateway, default false
	// +optional
	SSLEnabled bool
	// The name of the ScaleIO Protection Domain for the configured storage.
	// +optional
	ProtectionDomain string
	// The ScaleIO Storage Pool associated with the protection domain.
	// +optional
	StoragePool string
	// Indicates whether the storage for a volume should be ThickProvisioned or ThinProvisioned.
	// Default is ThinProvisioned.
	// +optional
	StorageMode string
	// The name of a volume already created in the ScaleIO system
	// that is associated with this volume source.
	VolumeName string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs".
	// Default is "xfs".
	// +optional
	FSType string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// ScaleIOPersistentVolumeSource represents a persistent ScaleIO volume that can be defined
// by a an admin via a storage class, for instance.
type ScaleIOPersistentVolumeSource struct {
	// The host address of the ScaleIO API Gateway.
	Gateway string
	// The name of the storage system as configured in ScaleIO.
	System string
	// SecretRef references to the secret for ScaleIO user and other
	// sensitive information. If this is not provided, Login operation will fail.
	SecretRef *SecretReference
	// Flag to enable/disable SSL communication with Gateway, default false
	// +optional
	SSLEnabled bool
	// The name of the ScaleIO Protection Domain for the configured storage.
	// +optional
	ProtectionDomain string
	// The ScaleIO Storage Pool associated with the protection domain.
	// +optional
	StoragePool string
	// Indicates whether the storage for a volume should be ThickProvisioned or ThinProvisioned.
	// Default is ThinProvisioned.
	// +optional
	StorageMode string
	// The name of a volume created in the ScaleIO system
	// that is associated with this volume source.
	VolumeName string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs".
	// Default is "xfs".
	// +optional
	FSType string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
}

// Represents a StorageOS persistent volume resource.
type StorageOSVolumeSource struct {
	// VolumeName is the human-readable name of the StorageOS volume.  Volume
	// names are only unique within a namespace.
	VolumeName string
	// VolumeNamespace specifies the scope of the volume within StorageOS.  If no
	// namespace is specified then the Pod's namespace will be used.  This allows the
	// Kubernetes name scoping to be mirrored within StorageOS for tighter integration.
	// Set VolumeName to any name to override the default behaviour.
	// Set to "default" if you are not using namespaces within StorageOS.
	// Namespaces that do not pre-exist within StorageOS will be created.
	// +optional
	VolumeNamespace string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// +optional
	FSType string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// SecretRef specifies the secret to use for obtaining the StorageOS API
	// credentials.  If not specified, default values will be attempted.
	// +optional
	SecretRef *LocalObjectReference
}

// Represents a StorageOS persistent volume resource.
type StorageOSPersistentVolumeSource struct {
	// VolumeName is the human-readable name of the StorageOS volume.  Volume
	// names are only unique within a namespace.
	VolumeName string
	// VolumeNamespace specifies the scope of the volume within StorageOS.  If no
	// namespace is specified then the Pod's namespace will be used.  This allows the
	// Kubernetes name scoping to be mirrored within StorageOS for tighter integration.
	// Set VolumeName to any name to override the default behaviour.
	// Set to "default" if you are not using namespaces within StorageOS.
	// Namespaces that do not pre-exist within StorageOS will be created.
	// +optional
	VolumeNamespace string
	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
	// +optional
	FSType string
	// Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	// +optional
	ReadOnly bool
	// SecretRef specifies the secret to use for obtaining the StorageOS API
	// credentials.  If not specified, default values will be attempted.
	// +optional
	SecretRef *ObjectReference
}

// Adapts a ConfigMap into a volume.
//
// The contents of the target ConfigMap's Data field will be presented in a
// volume as files using the keys in the Data field as the file names, unless
// the items element is populated with specific mappings of keys to paths.
// ConfigMap volumes support ownership management and SELinux relabeling.
type ConfigMapVolumeSource struct {
	LocalObjectReference
	// If unspecified, each key-value pair in the Data field of the referenced
	// ConfigMap will be projected into the volume as a file whose name is the
	// key and content is the value. If specified, the listed keys will be
	// projected into the specified paths, and unlisted keys will not be
	// present. If a key is specified which is not present in the ConfigMap,
	// the volume setup will error unless it is marked optional. Paths must be
	// relative and may not contain the '..' path or start with '..'.
	// +optional
	Items []KeyToPath
	// Mode bits to use on created files by default. Must be a value between
	// 0 and 0777.
	// Directories within the path are not affected by this setting.
	// This might be in conflict with other options that affect the file
	// mode, like fsGroup, and the result can be other mode bits set.
	// +optional
	DefaultMode *int32
	// Specify whether the ConfigMap or it's keys must be defined
	// +optional
	Optional *bool
}

// Adapts a ConfigMap into a projected volume.
//
// The contents of the target ConfigMap's Data field will be presented in a
// projected volume as files using the keys in the Data field as the file names,
// unless the items element is populated with specific mappings of keys to paths.
// Note that this is identical to a configmap volume source without the default
// mode.
type ConfigMapProjection struct {
	LocalObjectReference
	// If unspecified, each key-value pair in the Data field of the referenced
	// ConfigMap will be projected into the volume as a file whose name is the
	// key and content is the value. If specified, the listed keys will be
	// projected into the specified paths, and unlisted keys will not be
	// present. If a key is specified which is not present in the ConfigMap,
	// the volume setup will error unless it is marked optional. Paths must be
	// relative and may not contain the '..' path or start with '..'.
	// +optional
	Items []KeyToPath
	// Specify whether the ConfigMap or it's keys must be defined
	// +optional
	Optional *bool
}

// ServiceAccountTokenProjection represents a projected service account token
// volume. This projection can be used to insert a service account token into
// the pods runtime filesystem for use against APIs (Kubernetes API Server or
// otherwise).
type ServiceAccountTokenProjection struct {
	// Audience is the intended audience of the token. A recipient of a token
	// must identify itself with an identifier specified in the audience of the
	// token, and otherwise should reject the token. The audience defaults to the
	// identifier of the apiserver.
	Audience string
	// ExpirationSeconds is the requested duration of validity of the service
	// account token. As the token approaches expiration, the kubelet volume
	// plugin will proactively rotate the service account token. The kubelet will
	// start trying to rotate the token if the token is older than 80 percent of
	// its time to live or if the token is older than 24 hours.Defaults to 1 hour
	// and must be at least 10 minutes.
	ExpirationSeconds int64
	// Path is the path relative to the mount point of the file to project the
	// token into.
	Path string
}

// Represents a projected volume source
type ProjectedVolumeSource struct {
	// list of volume projections
	Sources []VolumeProjection
	// Mode bits to use on created files by default. Must be a value between
	// 0 and 0777.
	// Directories within the path are not affected by this setting.
	// This might be in conflict with other options that affect the file
	// mode, like fsGroup, and the result can be other mode bits set.
	// +optional
	DefaultMode *int32
}

// Projection that may be projected along with other supported volume types
type VolumeProjection struct {
	// all types below are the supported types for projection into the same volume

	// information about the secret data to project
	Secret *SecretProjection
	// information about the downwardAPI data to project
	DownwardAPI *DownwardAPIProjection
	// information about the configMap data to project
	ConfigMap *ConfigMapProjection
	// information about the serviceAccountToken data to project
	ServiceAccountToken *ServiceAccountTokenProjection
}

// Maps a string key to a path within a volume.
type KeyToPath struct {
	// The key to project.
	Key string

	// The relative path of the file to map the key to.
	// May not be an absolute path.
	// May not contain the path element '..'.
	// May not start with the string '..'.
	Path string
	// Optional: mode bits to use on this file, should be a value between 0
	// and 0777. If not specified, the volume defaultMode will be used.
	// This might be in conflict with other options that affect the file
	// mode, like fsGroup, and the result can be other mode bits set.
	// +optional
	Mode *int32
}

// Local represents directly-attached storage with node affinity (Beta feature)
type LocalVolumeSource struct {
	// The full path to the volume on the node.
	// It can be either a directory or block device (disk, partition, ...).
	Path string

	// Filesystem type to mount.
	// It applies only when the Path is a block device.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs". The default value is to auto-select a fileystem if unspecified.
	// +optional
	FSType *string
}

// Represents storage that is managed by an external CSI volume driver (Beta feature)
type CSIPersistentVolumeSource struct {
	// Driver is the name of the driver to use for this volume.
	// Required.
	Driver string

	// VolumeHandle is the unique volume name returned by the CSI volume
	// plugin’s CreateVolume to refer to the volume on all subsequent calls.
	// Required.
	VolumeHandle string

	// Optional: The value to pass to ControllerPublishVolumeRequest.
	// Defaults to false (read/write).
	// +optional
	ReadOnly bool

	// Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs".
	// +optional
	FSType string

	// Attributes of the volume to publish.
	// +optional
	VolumeAttributes map[string]string

	// ControllerPublishSecretRef is a reference to the secret object containing
	// sensitive information to pass to the CSI driver to complete the CSI
	// ControllerPublishVolume and ControllerUnpublishVolume calls.
	// This field is optional, and  may be empty if no secret is required. If the
	// secret object contains more than one secret, all secrets are passed.
	// +optional
	ControllerPublishSecretRef *SecretReference

	// NodeStageSecretRef is a reference to the secret object containing sensitive
	// information to pass to the CSI driver to complete the CSI NodeStageVolume
	// and NodeStageVolume and NodeUnstageVolume calls.
	// This field is optional, and  may be empty if no secret is required. If the
	// secret object contains more than one secret, all secrets are passed.
	// +optional
	NodeStageSecretRef *SecretReference

	// NodePublishSecretRef is a reference to the secret object containing
	// sensitive information to pass to the CSI driver to complete the CSI
	// NodePublishVolume and NodeUnpublishVolume calls.
	// This field is optional, and  may be empty if no secret is required. If the
	// secret object contains more than one secret, all secrets are passed.
	// +optional
	NodePublishSecretRef *SecretReference
}

// ContainerPort represents a network port in a single container
type ContainerPort struct {
	// Optional: If specified, this must be an IANA_SVC_NAME  Each named port
	// in a pod must have a unique name.
	// +optional
	Name string
	// Optional: If specified, this must be a valid port number, 0 < x < 65536.
	// If HostNetwork is specified, this must match ContainerPort.
	// +optional
	HostPort int32
	// Required: This must be a valid port number, 0 < x < 65536.
	ContainerPort int32
	// Required: Supports "TCP", "UDP" and "SCTP"
	// +optional
	Protocol Protocol
	// Optional: What host IP to bind the external port to.
	// +optional
	HostIP string
}

// VolumeMount describes a mounting of a Volume within a container.
type VolumeMount struct {
	// Required: This must match the Name of a Volume [above].
	Name string
	// Optional: Defaults to false (read-write).
	// +optional
	ReadOnly bool
	// Required. If the path is not an absolute path (e.g. some/path) it
	// will be prepended with the appropriate root prefix for the operating
	// system.  On Linux this is '/', on Windows this is 'C:\'.
	MountPath string
	// Path within the volume from which the container's volume should be mounted.
	// Defaults to "" (volume's root).
	// +optional
	SubPath string
	// mountPropagation determines how mounts are propagated from the host
	// to container and the other way around.
	// When not set, MountPropagationNone is used.
	// This field is beta in 1.10.
	// +optional
	MountPropagation *MountPropagationMode
}

// MountPropagationMode describes mount propagation.
type MountPropagationMode string

const (
	// MountPropagationNone means that the volume in a container will
	// not receive new mounts from the host or other containers, and filesystems
	// mounted inside the container won't be propagated to the host or other
	// containers.
	// Note that this mode corresponds to "private" in Linux terminology.
	MountPropagationNone MountPropagationMode = "None"
	// MountPropagationHostToContainer means that the volume in a container will
	// receive new mounts from the host or other containers, but filesystems
	// mounted inside the container won't be propagated to the host or other
	// containers.
	// Note that this mode is recursively applied to all mounts in the volume
	// ("rslave" in Linux terminology).
	MountPropagationHostToContainer MountPropagationMode = "HostToContainer"
	// MountPropagationBidirectional means that the volume in a container will
	// receive new mounts from the host or other containers, and its own mounts
	// will be propagated from the container to the host or other containers.
	// Note that this mode is recursively applied to all mounts in the volume
	// ("rshared" in Linux terminology).
	MountPropagationBidirectional MountPropagationMode = "Bidirectional"
)

// VolumeDevice describes a mapping of a raw block device within a container.
type VolumeDevice struct {
	// name must match the name of a persistentVolumeClaim in the pod
	Name string
	// devicePath is the path inside of the container that the device will be mapped to.
	DevicePath string
}

// EnvVar represents an environment variable present in a Container.
type EnvVar struct {
	// Required: This must be a C_IDENTIFIER.
	Name string
	// Optional: no more than one of the following may be specified.
	// Optional: Defaults to ""; variable references $(VAR_NAME) are expanded
	// using the previous defined environment variables in the container and
	// any service environment variables.  If a variable cannot be resolved,
	// the reference in the input string will be unchanged.  The $(VAR_NAME)
	// syntax can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped
	// references will never be expanded, regardless of whether the variable
	// exists or not.
	// +optional
	Value string
	// Optional: Specifies a source the value of this var should come from.
	// +optional
	ValueFrom *EnvVarSource
}

// EnvVarSource represents a source for the value of an EnvVar.
// Only one of its fields may be set.
type EnvVarSource struct {
	// Selects a field of the pod: supports metadata.name, metadata.namespace, metadata.labels, metadata.annotations,
	// metadata.uid, spec.nodeName, spec.serviceAccountName, status.hostIP, status.podIP.
	// +optional
	FieldRef *ObjectFieldSelector
	// Selects a resource of the container: only resources limits and requests
	// (limits.cpu, limits.memory, limits.ephemeral-storage, requests.cpu, requests.memory and requests.ephemeral-storage) are currently supported.
	// +optional
	ResourceFieldRef *ResourceFieldSelector
	// Selects a key of a ConfigMap.
	// +optional
	ConfigMapKeyRef *ConfigMapKeySelector
	// Selects a key of a secret in the pod's namespace.
	// +optional
	SecretKeyRef *SecretKeySelector
}

// ObjectFieldSelector selects an APIVersioned field of an object.
type ObjectFieldSelector struct {
	// Required: Version of the schema the FieldPath is written in terms of.
	// If no value is specified, it will be defaulted to the APIVersion of the
	// enclosing object.
	APIVersion string
	// Required: Path of the field to select in the specified API version
	FieldPath string
}

// ResourceFieldSelector represents container resources (cpu, memory) and their output format
type ResourceFieldSelector struct {
	// Container name: required for volumes, optional for env vars
	// +optional
	ContainerName string
	// Required: resource to select
	Resource string
	// Specifies the output format of the exposed resources, defaults to "1"
	// +optional
	Divisor resource.Quantity
}

// Selects a key from a ConfigMap.
type ConfigMapKeySelector struct {
	// The ConfigMap to select from.
	LocalObjectReference
	// The key to select.
	Key string
	// Specify whether the ConfigMap or it's key must be defined
	// +optional
	Optional *bool
}

// SecretKeySelector selects a key of a Secret.
type SecretKeySelector struct {
	// The name of the secret in the pod's namespace to select from.
	LocalObjectReference
	// The key of the secret to select from.  Must be a valid secret key.
	Key string
	// Specify whether the Secret or it's key must be defined
	// +optional
	Optional *bool
}

// EnvFromSource represents the source of a set of ConfigMaps
type EnvFromSource struct {
	// An optional identifier to prepend to each key in the ConfigMap.
	// +optional
	Prefix string
	// The ConfigMap to select from.
	//+optional
	ConfigMapRef *ConfigMapEnvSource
	// The Secret to select from.
	//+optional
	SecretRef *SecretEnvSource
}

// ConfigMapEnvSource selects a ConfigMap to populate the environment
// variables with.
//
// The contents of the target ConfigMap's Data field will represent the
// key-value pairs as environment variables.
type ConfigMapEnvSource struct {
	// The ConfigMap to select from.
	LocalObjectReference
	// Specify whether the ConfigMap must be defined
	// +optional
	Optional *bool
}

// SecretEnvSource selects a Secret to populate the environment
// variables with.
//
// The contents of the target Secret's Data field will represent the
// key-value pairs as environment variables.
type SecretEnvSource struct {
	// The Secret to select from.
	LocalObjectReference
	// Specify whether the Secret must be defined
	// +optional
	Optional *bool
}

// HTTPHeader describes a custom header to be used in HTTP probes
type HTTPHeader struct {
	// The header field name
	Name string
	// The header field value
	Value string
}

// HTTPGetAction describes an action based on HTTP Get requests.
type HTTPGetAction struct {
	// Optional: Path to access on the HTTP server.
	// +optional
	Path string
	// Required: Name or number of the port to access on the container.
	// +optional
	Port intstr.IntOrString
	// Optional: Host name to connect to, defaults to the pod IP. You
	// probably want to set "Host" in httpHeaders instead.
	// +optional
	Host string
	// Optional: Scheme to use for connecting to the host, defaults to HTTP.
	// +optional
	Scheme URIScheme
	// Optional: Custom headers to set in the request. HTTP allows repeated headers.
	// +optional
	HTTPHeaders []HTTPHeader
}

// URIScheme identifies the scheme used for connection to a host for Get actions
type URIScheme string

const (
	// URISchemeHTTP means that the scheme used will be http://
	URISchemeHTTP URIScheme = "HTTP"
	// URISchemeHTTPS means that the scheme used will be https://
	URISchemeHTTPS URIScheme = "HTTPS"
)

// TCPSocketAction describes an action based on opening a socket
type TCPSocketAction struct {
	// Required: Port to connect to.
	// +optional
	Port intstr.IntOrString
	// Optional: Host name to connect to, defaults to the pod IP.
	// +optional
	Host string
}

// ExecAction describes a "run in container" action.
type ExecAction struct {
	// Command is the command line to execute inside the container, the working directory for the
	// command  is root ('/') in the container's filesystem.  The command is simply exec'd, it is
	// not run inside a shell, so traditional shell instructions ('|', etc) won't work.  To use
	// a shell, you need to explicitly call out to that shell.
	// +optional
	Command []string
}

// Probe describes a health check to be performed against a container to determine whether it is
// alive or ready to receive traffic.
type Probe struct {
	// The action taken to determine the health of a container
	Handler
	// Length of time before health checking is activated.  In seconds.
	// +optional
	InitialDelaySeconds int32
	// Length of time before health checking times out.  In seconds.
	// +optional
	TimeoutSeconds int32
	// How often (in seconds) to perform the probe.
	// +optional
	PeriodSeconds int32
	// Minimum consecutive successes for the probe to be considered successful after having failed.
	// Must be 1 for liveness.
	// +optional
	SuccessThreshold int32
	// Minimum consecutive failures for the probe to be considered failed after having succeeded.
	// +optional
	FailureThreshold int32
}

// PullPolicy describes a policy for if/when to pull a container image
type PullPolicy string

const (
	// PullAlways means that kubelet always attempts to pull the latest image.  Container will fail If the pull fails.
	PullAlways PullPolicy = "Always"
	// PullNever means that kubelet never pulls an image, but only uses a local image.  Container will fail if the image isn't present
	PullNever PullPolicy = "Never"
	// PullIfNotPresent means that kubelet pulls if the image isn't present on disk. Container will fail if the image isn't present and the pull fails.
	PullIfNotPresent PullPolicy = "IfNotPresent"
)

// TerminationMessagePolicy describes how termination messages are retrieved from a container.
type TerminationMessagePolicy string

const (
	// TerminationMessageReadFile is the default behavior and will set the container status message to
	// the contents of the container's terminationMessagePath when the container exits.
	TerminationMessageReadFile TerminationMessagePolicy = "File"
	// TerminationMessageFallbackToLogsOnError will read the most recent contents of the container logs
	// for the container status message when the container exits with an error and the
	// terminationMessagePath has no contents.
	TerminationMessageFallbackToLogsOnError TerminationMessagePolicy = "FallbackToLogsOnError"
)

// Capability represent POSIX capabilities type
type Capability string

// Capabilities represent POSIX capabilities that can be added or removed to a running container.
type Capabilities struct {
	// Added capabilities
	// +optional
	Add []Capability
	// Removed capabilities
	// +optional
	Drop []Capability
}

// ResourceRequirements describes the compute resource requirements.
type ResourceRequirements struct {
	// Limits describes the maximum amount of compute resources allowed.
	// +optional
	Limits ResourceList
	// Requests describes the minimum amount of compute resources required.
	// If Request is omitted for a container, it defaults to Limits if that is explicitly specified,
	// otherwise to an implementation-defined value
	// +optional
	Requests ResourceList
}

// Container represents a single container that is expected to be run on the host.
type Container struct {
	// Required: This must be a DNS_LABEL.  Each container in a pod must
	// have a unique name.
	Name string
	// Required.
	Image string
	// Optional: The docker image's entrypoint is used if this is not provided; cannot be updated.
	// Variable references $(VAR_NAME) are expanded using the container's environment.  If a variable
	// cannot be resolved, the reference in the input string will be unchanged.  The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	// +optional
	Command []string
	// Optional: The docker image's cmd is used if this is not provided; cannot be updated.
	// Variable references $(VAR_NAME) are expanded using the container's environment.  If a variable
	// cannot be resolved, the reference in the input string will be unchanged.  The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	// +optional
	Args []string
	// Optional: Defaults to Docker's default.
	// +optional
	WorkingDir string
	// +optional
	Ports []ContainerPort
	// List of sources to populate environment variables in the container.
	// The keys defined within a source must be a C_IDENTIFIER. All invalid keys
	// will be reported as an event when the container is starting. When a key exists in multiple
	// sources, the value associated with the last source will take precedence.
	// Values defined by an Env with a duplicate key will take precedence.
	// Cannot be updated.
	// +optional
	EnvFrom []EnvFromSource
	// +optional
	Env []EnvVar
	// Compute resource requirements.
	// +optional
	Resources ResourceRequirements
	// +optional
	VolumeMounts []VolumeMount
	// volumeDevices is the list of block devices to be used by the container.
	// This is an alpha feature and may change in the future.
	// +optional
	VolumeDevices []VolumeDevice
	// +optional
	LivenessProbe *Probe
	// +optional
	ReadinessProbe *Probe
	// +optional
	Lifecycle *Lifecycle
	// Required.
	// +optional
	TerminationMessagePath string
	// +optional
	TerminationMessagePolicy TerminationMessagePolicy
	// Required: Policy for pulling images for this container
	ImagePullPolicy PullPolicy
	// Optional: SecurityContext defines the security options the container should be run with.
	// If set, the fields of SecurityContext override the equivalent fields of PodSecurityContext.
	// +optional
	SecurityContext *SecurityContext

	// Variables for interactive containers, these have very specialized use-cases (e.g. debugging)
	// and shouldn't be used for general purpose containers.
	// +optional
	Stdin bool
	// +optional
	StdinOnce bool
	// +optional
	TTY bool
}

// Handler defines a specific action that should be taken
// TODO: pass structured data to these actions, and document that data here.
type Handler struct {
	// One and only one of the following should be specified.
	// Exec specifies the action to take.
	// +optional
	Exec *ExecAction
	// HTTPGet specifies the http request to perform.
	// +optional
	HTTPGet *HTTPGetAction
	// TCPSocket specifies an action involving a TCP port.
	// TODO: implement a realistic TCP lifecycle hook
	// +optional
	TCPSocket *TCPSocketAction
}

// Lifecycle describes actions that the management system should take in response to container lifecycle
// events.  For the PostStart and PreStop lifecycle handlers, management of the container blocks
// until the action is complete, unless the container process fails, in which case the handler is aborted.
type Lifecycle struct {
	// PostStart is called immediately after a container is created.  If the handler fails, the container
	// is terminated and restarted.
	// +optional
	PostStart *Handler
	// PreStop is called immediately before a container is terminated.  The reason for termination is
	// passed to the handler.  Regardless of the outcome of the handler, the container is eventually terminated.
	// +optional
	PreStop *Handler
}

// The below types are used by kube_client and api_server.

type ConditionStatus string

// These are valid condition statuses. "ConditionTrue" means a resource is in the condition;
// "ConditionFalse" means a resource is not in the condition; "ConditionUnknown" means kubernetes
// can't decide if a resource is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

type ContainerStateWaiting struct {
	// A brief CamelCase string indicating details about why the container is in waiting state.
	// +optional
	Reason string
	// A human-readable message indicating details about why the container is in waiting state.
	// +optional
	Message string
}

type ContainerStateRunning struct {
	// +optional
	StartedAt metav1.Time
}

type ContainerStateTerminated struct {
	ExitCode int32
	// +optional
	Signal int32
	// +optional
	Reason string
	// +optional
	Message string
	// +optional
	StartedAt metav1.Time
	// +optional
	FinishedAt metav1.Time
	// +optional
	ContainerID string
}

// ContainerState holds a possible state of container.
// Only one of its members may be specified.
// If none of them is specified, the default one is ContainerStateWaiting.
type ContainerState struct {
	// +optional
	Waiting *ContainerStateWaiting
	// +optional
	Running *ContainerStateRunning
	// +optional
	Terminated *ContainerStateTerminated
}

type ContainerStatus struct {
	// Each container in a pod must have a unique name.
	Name string
	// +optional
	State ContainerState
	// +optional
	LastTerminationState ContainerState
	// Ready specifies whether the container has passed its readiness check.
	Ready bool
	// Note that this is calculated from dead containers.  But those containers are subject to
	// garbage collection.  This value will get capped at 5 by GC.
	RestartCount int32
	Image        string
	ImageID      string
	// +optional
	ContainerID string
}

// PodPhase is a label for the condition of a pod at the current time.
type PodPhase string

// These are the valid statuses of pods.
const (
	// PodPending means the pod has been accepted by the system, but one or more of the containers
	// has not been started. This includes time before being bound to a node, as well as time spent
	// pulling images onto the host.
	PodPending PodPhase = "Pending"
	// PodRunning means the pod has been bound to a node and all of the containers have been started.
	// At least one container is still running or is in the process of being restarted.
	PodRunning PodPhase = "Running"
	// PodSucceeded means that all containers in the pod have voluntarily terminated
	// with a container exit code of 0, and the system is not going to restart any of these containers.
	PodSucceeded PodPhase = "Succeeded"
	// PodFailed means that all containers in the pod have terminated, and at least one container has
	// terminated in a failure (exited with a non-zero exit code or was stopped by the system).
	PodFailed PodPhase = "Failed"
	// PodUnknown means that for some reason the state of the pod could not be obtained, typically due
	// to an error in communicating with the host of the pod.
	PodUnknown PodPhase = "Unknown"
)

type PodConditionType string

// These are valid conditions of pod.
const (
	// PodScheduled represents status of the scheduling process for this pod.
	PodScheduled PodConditionType = "PodScheduled"
	// PodReady means the pod is able to service requests and should be added to the
	// load balancing pools of all matching services.
	PodReady PodConditionType = "Ready"
	// PodInitialized means that all init containers in the pod have started successfully.
	PodInitialized PodConditionType = "Initialized"
	// PodReasonUnschedulable reason in PodScheduled PodCondition means that the scheduler
	// can't schedule the pod right now, for example due to insufficient resources in the cluster.
	PodReasonUnschedulable = "Unschedulable"
	// ContainersReady indicates whether all containers in the pod are ready.
	ContainersReady PodConditionType = "ContainersReady"
)

type PodCondition struct {
	Type   PodConditionType
	Status ConditionStatus
	// +optional
	LastProbeTime metav1.Time
	// +optional
	LastTransitionTime metav1.Time
	// +optional
	Reason string
	// +optional
	Message string
}

// RestartPolicy describes how the container should be restarted.
// Only one of the following restart policies may be specified.
// If none of the following policies is specified, the default one
// is RestartPolicyAlways.
type RestartPolicy string

const (
	RestartPolicyAlways    RestartPolicy = "Always"
	RestartPolicyOnFailure RestartPolicy = "OnFailure"
	RestartPolicyNever     RestartPolicy = "Never"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodList is a list of Pods.
type PodList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Pod
}

// DNSPolicy defines how a pod's DNS will be configured.
type DNSPolicy string

const (
	// DNSClusterFirstWithHostNet indicates that the pod should use cluster DNS
	// first, if it is available, then fall back on the default
	// (as determined by kubelet) DNS settings.
	DNSClusterFirstWithHostNet DNSPolicy = "ClusterFirstWithHostNet"

	// DNSClusterFirst indicates that the pod should use cluster DNS
	// first unless hostNetwork is true, if it is available, then
	// fall back on the default (as determined by kubelet) DNS settings.
	DNSClusterFirst DNSPolicy = "ClusterFirst"

	// DNSDefault indicates that the pod should use the default (as
	// determined by kubelet) DNS settings.
	DNSDefault DNSPolicy = "Default"

	// DNSNone indicates that the pod should use empty DNS settings. DNS
	// parameters such as nameservers and search paths should be defined via
	// DNSConfig.
	DNSNone DNSPolicy = "None"
)

// A node selector represents the union of the results of one or more label queries
// over a set of nodes; that is, it represents the OR of the selectors represented
// by the node selector terms.
type NodeSelector struct {
	//Required. A list of node selector terms. The terms are ORed.
	NodeSelectorTerms []NodeSelectorTerm
}

// A null or empty node selector term matches no objects. The requirements of
// them are ANDed.
// The TopologySelectorTerm type implements a subset of the NodeSelectorTerm.
type NodeSelectorTerm struct {
	// A list of node selector requirements by node's labels.
	MatchExpressions []NodeSelectorRequirement
	// A list of node selector requirements by node's fields.
	MatchFields []NodeSelectorRequirement
}

// A node selector requirement is a selector that contains values, a key, and an operator
// that relates the key and values.
type NodeSelectorRequirement struct {
	// The label key that the selector applies to.
	Key string
	// Represents a key's relationship to a set of values.
	// Valid operators are In, NotIn, Exists, DoesNotExist. Gt, and Lt.
	Operator NodeSelectorOperator
	// An array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty. If the operator is Gt or Lt, the values
	// array must have a single element, which will be interpreted as an integer.
	// This array is replaced during a strategic merge patch.
	// +optional
	Values []string
}

// A node selector operator is the set of operators that can be used in
// a node selector requirement.
type NodeSelectorOperator string

const (
	NodeSelectorOpIn           NodeSelectorOperator = "In"
	NodeSelectorOpNotIn        NodeSelectorOperator = "NotIn"
	NodeSelectorOpExists       NodeSelectorOperator = "Exists"
	NodeSelectorOpDoesNotExist NodeSelectorOperator = "DoesNotExist"
	NodeSelectorOpGt           NodeSelectorOperator = "Gt"
	NodeSelectorOpLt           NodeSelectorOperator = "Lt"
)

// A topology selector term represents the result of label queries.
// A null or empty topology selector term matches no objects.
// The requirements of them are ANDed.
// It provides a subset of functionality as NodeSelectorTerm.
// This is an alpha feature and may change in the future.
type TopologySelectorTerm struct {
	// A list of topology selector requirements by labels.
	// +optional
	MatchLabelExpressions []TopologySelectorLabelRequirement
}

// A topology selector requirement is a selector that matches given label.
// This is an alpha feature and may change in the future.
type TopologySelectorLabelRequirement struct {
	// The label key that the selector applies to.
	Key string
	// An array of string values. One value must match the label to be selected.
	// Each entry in Values is ORed.
	Values []string
}

// Affinity is a group of affinity scheduling rules.
type Affinity struct {
	// Describes node affinity scheduling rules for the pod.
	// +optional
	NodeAffinity *NodeAffinity
	// Describes pod affinity scheduling rules (e.g. co-locate this pod in the same node, zone, etc. as some other pod(s)).
	// +optional
	PodAffinity *PodAffinity
	// Describes pod anti-affinity scheduling rules (e.g. avoid putting this pod in the same node, zone, etc. as some other pod(s)).
	// +optional
	PodAntiAffinity *PodAntiAffinity
}

// Pod affinity is a group of inter pod affinity scheduling rules.
type PodAffinity struct {
	// NOT YET IMPLEMENTED. TODO: Uncomment field once it is implemented.
	// If the affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to a pod label update), the
	// system will try to eventually evict the pod from its node.
	// When there are multiple elements, the lists of nodes corresponding to each
	// podAffinityTerm are intersected, i.e. all terms must be satisfied.
	// +optional
	// RequiredDuringSchedulingRequiredDuringExecution []PodAffinityTerm

	// If the affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to a pod label update), the
	// system may or may not try to eventually evict the pod from its node.
	// When there are multiple elements, the lists of nodes corresponding to each
	// podAffinityTerm are intersected, i.e. all terms must be satisfied.
	// +optional
	RequiredDuringSchedulingIgnoredDuringExecution []PodAffinityTerm
	// The scheduler will prefer to schedule pods to nodes that satisfy
	// the affinity expressions specified by this field, but it may choose
	// a node that violates one or more of the expressions. The node that is
	// most preferred is the one with the greatest sum of weights, i.e.
	// for each node that meets all of the scheduling requirements (resource
	// request, requiredDuringScheduling affinity expressions, etc.),
	// compute a sum by iterating through the elements of this field and adding
	// "weight" to the sum if the node has pods which matches the corresponding podAffinityTerm; the
	// node(s) with the highest sum are the most preferred.
	// +optional
	PreferredDuringSchedulingIgnoredDuringExecution []WeightedPodAffinityTerm
}

// Pod anti affinity is a group of inter pod anti affinity scheduling rules.
type PodAntiAffinity struct {
	// NOT YET IMPLEMENTED. TODO: Uncomment field once it is implemented.
	// If the anti-affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the anti-affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to a pod label update), the
	// system will try to eventually evict the pod from its node.
	// When there are multiple elements, the lists of nodes corresponding to each
	// podAffinityTerm are intersected, i.e. all terms must be satisfied.
	// +optional
	// RequiredDuringSchedulingRequiredDuringExecution []PodAffinityTerm

	// If the anti-affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the anti-affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to a pod label update), the
	// system may or may not try to eventually evict the pod from its node.
	// When there are multiple elements, the lists of nodes corresponding to each
	// podAffinityTerm are intersected, i.e. all terms must be satisfied.
	// +optional
	RequiredDuringSchedulingIgnoredDuringExecution []PodAffinityTerm
	// The scheduler will prefer to schedule pods to nodes that satisfy
	// the anti-affinity expressions specified by this field, but it may choose
	// a node that violates one or more of the expressions. The node that is
	// most preferred is the one with the greatest sum of weights, i.e.
	// for each node that meets all of the scheduling requirements (resource
	// request, requiredDuringScheduling anti-affinity expressions, etc.),
	// compute a sum by iterating through the elements of this field and adding
	// "weight" to the sum if the node has pods which matches the corresponding podAffinityTerm; the
	// node(s) with the highest sum are the most preferred.
	// +optional
	PreferredDuringSchedulingIgnoredDuringExecution []WeightedPodAffinityTerm
}

// The weights of all of the matched WeightedPodAffinityTerm fields are added per-node to find the most preferred node(s)
type WeightedPodAffinityTerm struct {
	// weight associated with matching the corresponding podAffinityTerm,
	// in the range 1-100.
	Weight int32
	// Required. A pod affinity term, associated with the corresponding weight.
	PodAffinityTerm PodAffinityTerm
}

// Defines a set of pods (namely those matching the labelSelector
// relative to the given namespace(s)) that this pod should be
// co-located (affinity) or not co-located (anti-affinity) with,
// where co-located is defined as running on a node whose value of
// the label with key <topologyKey> matches that of any node on which
// a pod of the set of pods is running.
type PodAffinityTerm struct {
	// A label query over a set of resources, in this case pods.
	// +optional
	LabelSelector *metav1.LabelSelector
	// namespaces specifies which namespaces the labelSelector applies to (matches against);
	// null or empty list means "this pod's namespace"
	// +optional
	Namespaces []string
	// This pod should be co-located (affinity) or not co-located (anti-affinity) with the pods matching
	// the labelSelector in the specified namespaces, where co-located is defined as running on a node
	// whose value of the label with key topologyKey matches that of any node on which any of the
	// selected pods is running.
	// Empty topologyKey is not allowed.
	TopologyKey string
}

// Node affinity is a group of node affinity scheduling rules.
type NodeAffinity struct {
	// NOT YET IMPLEMENTED. TODO: Uncomment field once it is implemented.
	// If the affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to an update), the system
	// will try to eventually evict the pod from its node.
	// +optional
	// RequiredDuringSchedulingRequiredDuringExecution *NodeSelector

	// If the affinity requirements specified by this field are not met at
	// scheduling time, the pod will not be scheduled onto the node.
	// If the affinity requirements specified by this field cease to be met
	// at some point during pod execution (e.g. due to an update), the system
	// may or may not try to eventually evict the pod from its node.
	// +optional
	RequiredDuringSchedulingIgnoredDuringExecution *NodeSelector
	// The scheduler will prefer to schedule pods to nodes that satisfy
	// the affinity expressions specified by this field, but it may choose
	// a node that violates one or more of the expressions. The node that is
	// most preferred is the one with the greatest sum of weights, i.e.
	// for each node that meets all of the scheduling requirements (resource
	// request, requiredDuringScheduling affinity expressions, etc.),
	// compute a sum by iterating through the elements of this field and adding
	// "weight" to the sum if the node matches the corresponding matchExpressions; the
	// node(s) with the highest sum are the most preferred.
	// +optional
	PreferredDuringSchedulingIgnoredDuringExecution []PreferredSchedulingTerm
}

// An empty preferred scheduling term matches all objects with implicit weight 0
// (i.e. it's a no-op). A null preferred scheduling term matches no objects (i.e. is also a no-op).
type PreferredSchedulingTerm struct {
	// Weight associated with matching the corresponding nodeSelectorTerm, in the range 1-100.
	Weight int32
	// A node selector term, associated with the corresponding weight.
	Preference NodeSelectorTerm
}

// The node this Taint is attached to has the "effect" on
// any pod that does not tolerate the Taint.
type Taint struct {
	// Required. The taint key to be applied to a node.
	Key string
	// Required. The taint value corresponding to the taint key.
	// +optional
	Value string
	// Required. The effect of the taint on pods
	// that do not tolerate the taint.
	// Valid effects are NoSchedule, PreferNoSchedule and NoExecute.
	Effect TaintEffect
	// TimeAdded represents the time at which the taint was added.
	// It is only written for NoExecute taints.
	// +optional
	TimeAdded *metav1.Time
}

type TaintEffect string

const (
	// Do not allow new pods to schedule onto the node unless they tolerate the taint,
	// but allow all pods submitted to Kubelet without going through the scheduler
	// to start, and allow all already-running pods to continue running.
	// Enforced by the scheduler.
	TaintEffectNoSchedule TaintEffect = "NoSchedule"
	// Like TaintEffectNoSchedule, but the scheduler tries not to schedule
	// new pods onto the node, rather than prohibiting new pods from scheduling
	// onto the node entirely. Enforced by the scheduler.
	TaintEffectPreferNoSchedule TaintEffect = "PreferNoSchedule"
	// NOT YET IMPLEMENTED. TODO: Uncomment field once it is implemented.
	// Like TaintEffectNoSchedule, but additionally do not allow pods submitted to
	// Kubelet without going through the scheduler to start.
	// Enforced by Kubelet and the scheduler.
	// TaintEffectNoScheduleNoAdmit TaintEffect = "NoScheduleNoAdmit"

	// Evict any already-running pods that do not tolerate the taint.
	// Currently enforced by NodeController.
	TaintEffectNoExecute TaintEffect = "NoExecute"
)

// The pod this Toleration is attached to tolerates any taint that matches
// the triple <key,value,effect> using the matching operator <operator>.
type Toleration struct {
	// Key is the taint key that the toleration applies to. Empty means match all taint keys.
	// If the key is empty, operator must be Exists; this combination means to match all values and all keys.
	// +optional
	Key string
	// Operator represents a key's relationship to the value.
	// Valid operators are Exists and Equal. Defaults to Equal.
	// Exists is equivalent to wildcard for value, so that a pod can
	// tolerate all taints of a particular category.
	// +optional
	Operator TolerationOperator
	// Value is the taint value the toleration matches to.
	// If the operator is Exists, the value should be empty, otherwise just a regular string.
	// +optional
	Value string
	// Effect indicates the taint effect to match. Empty means match all taint effects.
	// When specified, allowed values are NoSchedule, PreferNoSchedule and NoExecute.
	// +optional
	Effect TaintEffect
	// TolerationSeconds represents the period of time the toleration (which must be
	// of effect NoExecute, otherwise this field is ignored) tolerates the taint. By default,
	// it is not set, which means tolerate the taint forever (do not evict). Zero and
	// negative values will be treated as 0 (evict immediately) by the system.
	// +optional
	TolerationSeconds *int64
}

// A toleration operator is the set of operators that can be used in a toleration.
type TolerationOperator string

const (
	TolerationOpExists TolerationOperator = "Exists"
	TolerationOpEqual  TolerationOperator = "Equal"
)

// PodReadinessGate contains the reference to a pod condition
type PodReadinessGate struct {
	// ConditionType refers to a condition in the pod's condition list with matching type.
	ConditionType PodConditionType
}

// PodSpec is a description of a pod
type PodSpec struct {
	Volumes []Volume
	// List of initialization containers belonging to the pod.
	InitContainers []Container
	// List of containers belonging to the pod.
	Containers []Container
	// +optional
	RestartPolicy RestartPolicy
	// Optional duration in seconds the pod needs to terminate gracefully. May be decreased in delete request.
	// Value must be non-negative integer. The value zero indicates delete immediately.
	// If this value is nil, the default grace period will be used instead.
	// The grace period is the duration in seconds after the processes running in the pod are sent
	// a termination signal and the time when the processes are forcibly halted with a kill signal.
	// Set this value longer than the expected cleanup time for your process.
	// +optional
	TerminationGracePeriodSeconds *int64
	// Optional duration in seconds relative to the StartTime that the pod may be active on a node
	// before the system actively tries to terminate the pod; value must be positive integer
	// +optional
	ActiveDeadlineSeconds *int64
	// Set DNS policy for the pod.
	// Defaults to "ClusterFirst".
	// Valid values are 'ClusterFirstWithHostNet', 'ClusterFirst', 'Default' or 'None'.
	// DNS parameters given in DNSConfig will be merged with the policy selected with DNSPolicy.
	// To have DNS options set along with hostNetwork, you have to specify DNS policy
	// explicitly to 'ClusterFirstWithHostNet'.
	// +optional
	DNSPolicy DNSPolicy
	// NodeSelector is a selector which must be true for the pod to fit on a node
	// +optional
	NodeSelector map[string]string

	// ServiceAccountName is the name of the ServiceAccount to use to run this pod
	// The pod will be allowed to use secrets referenced by the ServiceAccount
	ServiceAccountName string
	// AutomountServiceAccountToken indicates whether a service account token should be automatically mounted.
	// +optional
	AutomountServiceAccountToken *bool

	// NodeName is a request to schedule this pod onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this pod onto that node, assuming that it fits resource
	// requirements.
	// +optional
	NodeName string
	// SecurityContext holds pod-level security attributes and common container settings.
	// Optional: Defaults to empty.  See type description for default values of each field.
	// +optional
	SecurityContext *PodSecurityContext
	// ImagePullSecrets is an optional list of references to secrets in the same namespace to use for pulling any of the images used by this PodSpec.
	// If specified, these secrets will be passed to individual puller implementations for them to use.  For example,
	// in the case of docker, only DockerConfig type secrets are honored.
	// +optional
	ImagePullSecrets []LocalObjectReference
	// Specifies the hostname of the Pod.
	// If not specified, the pod's hostname will be set to a system-defined value.
	// +optional
	Hostname string
	// If specified, the fully qualified Pod hostname will be "<hostname>.<subdomain>.<pod namespace>.svc.<cluster domain>".
	// If not specified, the pod will not have a domainname at all.
	// +optional
	Subdomain string
	// If specified, the pod's scheduling constraints
	// +optional
	Affinity *Affinity
	// If specified, the pod will be dispatched by specified scheduler.
	// If not specified, the pod will be dispatched by default scheduler.
	// +optional
	SchedulerName string
	// If specified, the pod's tolerations.
	// +optional
	Tolerations []Toleration
	// HostAliases is an optional list of hosts and IPs that will be injected into the pod's hosts
	// file if specified. This is only valid for non-hostNetwork pods.
	// +optional
	HostAliases []HostAlias
	// If specified, indicates the pod's priority. "system-node-critical" and
	// "system-cluster-critical" are two special keywords which indicate the
	// highest priorities with the former being the highest priority. Any other
	// name must be defined by creating a PriorityClass object with that name.
	// If not specified, the pod priority will be default or zero if there is no
	// default.
	// +optional
	PriorityClassName string
	// The priority value. Various system components use this field to find the
	// priority of the pod. When Priority Admission Controller is enabled, it
	// prevents users from setting this field. The admission controller populates
	// this field from PriorityClassName.
	// The higher the value, the higher the priority.
	// +optional
	Priority *int32
	// Specifies the DNS parameters of a pod.
	// Parameters specified here will be merged to the generated DNS
	// configuration based on DNSPolicy.
	// +optional
	DNSConfig *PodDNSConfig
	// If specified, all readiness gates will be evaluated for pod readiness.
	// A pod is ready when all its containers are ready AND
	// all conditions specified in the readiness gates have status equal to "True"
	// More info: https://github.com/kubernetes/community/blob/master/keps/sig-network/0007-pod-ready%2B%2B.md
	// +optional
	ReadinessGates []PodReadinessGate
	// RuntimeClassName refers to a RuntimeClass object in the node.k8s.io group, which should be used
	// to run this pod.  If no RuntimeClass resource matches the named class, the pod will not be run.
	// If unset or empty, the "legacy" RuntimeClass will be used, which is an implicit class with an
	// empty definition that uses the default runtime handler.
	// More info: https://github.com/kubernetes/community/blob/master/keps/sig-node/0014-runtime-class.md
	// This is an alpha feature and may change in the future.
	// +optional
	RuntimeClassName *string
	// EnableServiceLinks indicates whether information about services should be injected into pod's
	// environment variables, matching the syntax of Docker links.
	// If not specified, the default is true.
	// +optional
	EnableServiceLinks *bool
}

// HostAlias holds the mapping between IP and hostnames that will be injected as an entry in the
// pod's hosts file.
type HostAlias struct {
	IP        string
	Hostnames []string
}

// Sysctl defines a kernel parameter to be set
type Sysctl struct {
	// Name of a property to set
	Name string
	// Value of a property to set
	Value string
}

// PodSecurityContext holds pod-level security attributes and common container settings.
// Some fields are also present in container.securityContext.  Field values of
// container.securityContext take precedence over field values of PodSecurityContext.
type PodSecurityContext struct {
	// Use the host's network namespace.  If this option is set, the ports that will be
	// used must be specified.
	// Optional: Default to false
	// +k8s:conversion-gen=false
	// +optional
	HostNetwork bool
	// Use the host's pid namespace.
	// Optional: Default to false.
	// +k8s:conversion-gen=false
	// +optional
	HostPID bool
	// Use the host's ipc namespace.
	// Optional: Default to false.
	// +k8s:conversion-gen=false
	// +optional
	HostIPC bool
	// Share a single process namespace between all of the containers in a pod.
	// When this is set containers will be able to view and signal processes from other containers
	// in the same pod, and the first process in each container will not be assigned PID 1.
	// HostPID and ShareProcessNamespace cannot both be set.
	// Optional: Default to false.
	// This field is beta-level and may be disabled with the PodShareProcessNamespace feature.
	// +k8s:conversion-gen=false
	// +optional
	ShareProcessNamespace *bool
	// The SELinux context to be applied to all containers.
	// If unspecified, the container runtime will allocate a random SELinux context for each
	// container.  May also be set in SecurityContext.  If set in
	// both SecurityContext and PodSecurityContext, the value specified in SecurityContext
	// takes precedence for that container.
	// +optional
	SELinuxOptions *SELinuxOptions
	// The UID to run the entrypoint of the container process.
	// Defaults to user specified in image metadata if unspecified.
	// May also be set in SecurityContext.  If set in both SecurityContext and
	// PodSecurityContext, the value specified in SecurityContext takes precedence
	// for that container.
	// +optional
	RunAsUser *int64
	// The GID to run the entrypoint of the container process.
	// Uses runtime default if unset.
	// May also be set in SecurityContext.  If set in both SecurityContext and
	// PodSecurityContext, the value specified in SecurityContext takes precedence
	// for that container.
	// +optional
	RunAsGroup *int64
	// Indicates that the container must run as a non-root user.
	// If true, the Kubelet will validate the image at runtime to ensure that it
	// does not run as UID 0 (root) and fail to start the container if it does.
	// If unset or false, no such validation will be performed.
	// May also be set in SecurityContext.  If set in both SecurityContext and
	// PodSecurityContext, the value specified in SecurityContext takes precedence
	// for that container.
	// +optional
	RunAsNonRoot *bool
	// A list of groups applied to the first process run in each container, in addition
	// to the container's primary GID.  If unspecified, no groups will be added to
	// any container.
	// +optional
	SupplementalGroups []int64
	// A special supplemental group that applies to all containers in a pod.
	// Some volume types allow the Kubelet to change the ownership of that volume
	// to be owned by the pod:
	//
	// 1. The owning GID will be the FSGroup
	// 2. The setgid bit is set (new files created in the volume will be owned by FSGroup)
	// 3. The permission bits are OR'd with rw-rw----
	//
	// If unset, the Kubelet will not modify the ownership and permissions of any volume.
	// +optional
	FSGroup *int64
	// Sysctls hold a list of namespaced sysctls used for the pod. Pods with unsupported
	// sysctls (by the container runtime) might fail to launch.
	// +optional
	Sysctls []Sysctl
}

// PodQOSClass defines the supported qos classes of Pods.
type PodQOSClass string

const (
	// PodQOSGuaranteed is the Guaranteed qos class.
	PodQOSGuaranteed PodQOSClass = "Guaranteed"
	// PodQOSBurstable is the Burstable qos class.
	PodQOSBurstable PodQOSClass = "Burstable"
	// PodQOSBestEffort is the BestEffort qos class.
	PodQOSBestEffort PodQOSClass = "BestEffort"
)

// PodDNSConfig defines the DNS parameters of a pod in addition to
// those generated from DNSPolicy.
type PodDNSConfig struct {
	// A list of DNS name server IP addresses.
	// This will be appended to the base nameservers generated from DNSPolicy.
	// Duplicated nameservers will be removed.
	// +optional
	Nameservers []string
	// A list of DNS search domains for host-name lookup.
	// This will be appended to the base search paths generated from DNSPolicy.
	// Duplicated search paths will be removed.
	// +optional
	Searches []string
	// A list of DNS resolver options.
	// This will be merged with the base options generated from DNSPolicy.
	// Duplicated entries will be removed. Resolution options given in Options
	// will override those that appear in the base DNSPolicy.
	// +optional
	Options []PodDNSConfigOption
}

// PodDNSConfigOption defines DNS resolver options of a pod.
type PodDNSConfigOption struct {
	// Required.
	Name string
	// +optional
	Value *string
}

// PodStatus represents information about the status of a pod. Status may trail the actual
// state of a system.
type PodStatus struct {
	// +optional
	Phase PodPhase
	// +optional
	Conditions []PodCondition
	// A human readable message indicating details about why the pod is in this state.
	// +optional
	Message string
	// A brief CamelCase message indicating details about why the pod is in this state. e.g. 'Evicted'
	// +optional
	Reason string
	// nominatedNodeName is set when this pod preempts other pods on the node, but it cannot be
	// scheduled right away as preemption victims receive their graceful termination periods.
	// This field does not guarantee that the pod will be scheduled on this node. Scheduler may decide
	// to place the pod elsewhere if other nodes become available sooner. Scheduler may also decide to
	// give the resources on this node to a higher priority pod that is created after preemption.
	// +optional
	NominatedNodeName string

	// +optional
	HostIP string
	// +optional
	PodIP string

	// Date and time at which the object was acknowledged by the Kubelet.
	// This is before the Kubelet pulled the container image(s) for the pod.
	// +optional
	StartTime *metav1.Time
	// +optional
	QOSClass PodQOSClass

	// The list has one entry per init container in the manifest. The most recent successful
	// init container will have ready = true, the most recently started container will have
	// startTime set.
	// More info: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-and-container-status
	InitContainerStatuses []ContainerStatus
	// The list has one entry per container in the manifest. Each entry is
	// currently the output of `docker inspect`. This output format is *not*
	// final and should not be relied upon.
	// TODO: Make real decisions about what our info should look like. Re-enable fuzz test
	// when we have done this.
	// +optional
	ContainerStatuses []ContainerStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodStatusResult is a wrapper for PodStatus returned by kubelet that can be encode/decoded
type PodStatusResult struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta
	// Status represents the current information about a pod. This data may not be up
	// to date.
	// +optional
	Status PodStatus
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Pod is a collection of containers, used as either input (create, update) or as output (list, get).
type Pod struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the behavior of a pod.
	// +optional
	Spec PodSpec

	// Status represents the current information about a pod. This data may not be up
	// to date.
	// +optional
	Status PodStatus
}

// PodTemplateSpec describes the data a pod should have when created from a template
type PodTemplateSpec struct {
	// Metadata of the pods created from this template.
	// +optional
	metav1.ObjectMeta

	// Spec defines the behavior of a pod.
	// +optional
	Spec PodSpec
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodTemplate describes a template for creating copies of a predefined pod.
type PodTemplate struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Template defines the pods that will be created from this pod template
	// +optional
	Template PodTemplateSpec
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodTemplateList is a list of PodTemplates.
type PodTemplateList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []PodTemplate
}

// ReplicationControllerSpec is the specification of a replication controller.
// As the internal representation of a replication controller, it may have either
// a TemplateRef or a Template set.
type ReplicationControllerSpec struct {
	// Replicas is the number of desired replicas.
	Replicas int32

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	// +optional
	MinReadySeconds int32

	// Selector is a label query over pods that should match the Replicas count.
	Selector map[string]string

	// TemplateRef is a reference to an object that describes the pod that will be created if
	// insufficient replicas are detected. This reference is ignored if a Template is set.
	// Must be set before converting to a versioned API object
	// +optional
	//TemplateRef *ObjectReference

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected. Internally, this takes precedence over a
	// TemplateRef.
	// +optional
	Template *PodTemplateSpec
}

// ReplicationControllerStatus represents the current status of a replication
// controller.
type ReplicationControllerStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int32

	// The number of pods that have labels matching the labels of the pod template of the replication controller.
	// +optional
	FullyLabeledReplicas int32

	// The number of ready replicas for this replication controller.
	// +optional
	ReadyReplicas int32

	// The number of available replicas (ready for at least minReadySeconds) for this replication controller.
	// +optional
	AvailableReplicas int32

	// ObservedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64

	// Represents the latest available observations of a replication controller's current state.
	// +optional
	Conditions []ReplicationControllerCondition
}

type ReplicationControllerConditionType string

// These are valid conditions of a replication controller.
const (
	// ReplicationControllerReplicaFailure is added in a replication controller when one of its pods
	// fails to be created due to insufficient quota, limit ranges, pod security policy, node selectors,
	// etc. or deleted due to kubelet being down or finalizers are failing.
	ReplicationControllerReplicaFailure ReplicationControllerConditionType = "ReplicaFailure"
)

// ReplicationControllerCondition describes the state of a replication controller at a certain point.
type ReplicationControllerCondition struct {
	// Type of replication controller condition.
	Type ReplicationControllerConditionType
	// Status of the condition, one of True, False, Unknown.
	Status ConditionStatus
	// The last time the condition transitioned from one status to another.
	// +optional
	LastTransitionTime metav1.Time
	// The reason for the condition's last transition.
	// +optional
	Reason string
	// A human readable message indicating details about the transition.
	// +optional
	Message string
}

// +genclient
// +genclient:method=GetScale,verb=get,subresource=scale,result=k8s.io/kubernetes/pkg/apis/autoscaling.Scale
// +genclient:method=UpdateScale,verb=update,subresource=scale,input=k8s.io/kubernetes/pkg/apis/autoscaling.Scale,result=k8s.io/kubernetes/pkg/apis/autoscaling.Scale
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ReplicationController represents the configuration of a replication controller.
type ReplicationController struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the desired behavior of this replication controller.
	// +optional
	Spec ReplicationControllerSpec

	// Status is the current status of this replication controller. This data may be
	// out of date by some window of time.
	// +optional
	Status ReplicationControllerStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ReplicationControllerList is a collection of replication controllers.
type ReplicationControllerList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []ReplicationController
}

const (
	// ClusterIPNone - do not assign a cluster IP
	// no proxying required and no environment variables should be created for pods
	ClusterIPNone = "None"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceList holds a list of services.
type ServiceList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Service
}

// Session Affinity Type string
type ServiceAffinity string

const (
	// ServiceAffinityClientIP is the Client IP based.
	ServiceAffinityClientIP ServiceAffinity = "ClientIP"

	// ServiceAffinityNone - no session affinity.
	ServiceAffinityNone ServiceAffinity = "None"
)

const (
	// DefaultClientIPServiceAffinitySeconds is the default timeout seconds
	// of Client IP based session affinity - 3 hours.
	DefaultClientIPServiceAffinitySeconds int32 = 10800
	// MaxClientIPServiceAffinitySeconds is the max timeout seconds
	// of Client IP based session affinity - 1 day.
	MaxClientIPServiceAffinitySeconds int32 = 86400
)

// SessionAffinityConfig represents the configurations of session affinity.
type SessionAffinityConfig struct {
	// clientIP contains the configurations of Client IP based session affinity.
	// +optional
	ClientIP *ClientIPConfig
}

// ClientIPConfig represents the configurations of Client IP based session affinity.
type ClientIPConfig struct {
	// timeoutSeconds specifies the seconds of ClientIP type session sticky time.
	// The value must be >0 && <=86400(for 1 day) if ServiceAffinity == "ClientIP".
	// Default value is 10800(for 3 hours).
	// +optional
	TimeoutSeconds *int32
}

// Service Type string describes ingress methods for a service
type ServiceType string

const (
	// ServiceTypeClusterIP means a service will only be accessible inside the
	// cluster, via the ClusterIP.
	ServiceTypeClusterIP ServiceType = "ClusterIP"

	// ServiceTypeNodePort means a service will be exposed on one port of
	// every node, in addition to 'ClusterIP' type.
	ServiceTypeNodePort ServiceType = "NodePort"

	// ServiceTypeLoadBalancer means a service will be exposed via an
	// external load balancer (if the cloud provider supports it), in addition
	// to 'NodePort' type.
	ServiceTypeLoadBalancer ServiceType = "LoadBalancer"

	// ServiceTypeExternalName means a service consists of only a reference to
	// an external name that kubedns or equivalent will return as a CNAME
	// record, with no exposing or proxying of any pods involved.
	ServiceTypeExternalName ServiceType = "ExternalName"
)

// Service External Traffic Policy Type string
type ServiceExternalTrafficPolicyType string

const (
	// ServiceExternalTrafficPolicyTypeLocal specifies node-local endpoints behavior.
	ServiceExternalTrafficPolicyTypeLocal ServiceExternalTrafficPolicyType = "Local"
	// ServiceExternalTrafficPolicyTypeCluster specifies cluster-wide (legacy) behavior.
	ServiceExternalTrafficPolicyTypeCluster ServiceExternalTrafficPolicyType = "Cluster"
)

// ServiceStatus represents the current status of a service
type ServiceStatus struct {
	// LoadBalancer contains the current status of the load-balancer,
	// if one is present.
	// +optional
	LoadBalancer LoadBalancerStatus
}

// LoadBalancerStatus represents the status of a load-balancer
type LoadBalancerStatus struct {
	// Ingress is a list containing ingress points for the load-balancer;
	// traffic intended for the service should be sent to these ingress points.
	// +optional
	Ingress []LoadBalancerIngress
}

// LoadBalancerIngress represents the status of a load-balancer ingress point:
// traffic intended for the service should be sent to an ingress point.
type LoadBalancerIngress struct {
	// IP is set for load-balancer ingress points that are IP based
	// (typically GCE or OpenStack load-balancers)
	// +optional
	IP string

	// Hostname is set for load-balancer ingress points that are DNS based
	// (typically AWS load-balancers)
	// +optional
	Hostname string
}

// ServiceSpec describes the attributes that a user creates on a service
type ServiceSpec struct {
	// Type determines how the Service is exposed. Defaults to ClusterIP. Valid
	// options are ExternalName, ClusterIP, NodePort, and LoadBalancer.
	// "ExternalName" maps to the specified externalName.
	// "ClusterIP" allocates a cluster-internal IP address for load-balancing to
	// endpoints. Endpoints are determined by the selector or if that is not
	// specified, by manual construction of an Endpoints object. If clusterIP is
	// "None", no virtual IP is allocated and the endpoints are published as a
	// set of endpoints rather than a stable IP.
	// "NodePort" builds on ClusterIP and allocates a port on every node which
	// routes to the clusterIP.
	// "LoadBalancer" builds on NodePort and creates an
	// external load-balancer (if supported in the current cloud) which routes
	// to the clusterIP.
	// More info: https://kubernetes.io/docs/concepts/services-networking/service/
	// +optional
	Type ServiceType

	// Required: The list of ports that are exposed by this service.
	Ports []ServicePort

	// Route service traffic to pods with label keys and values matching this
	// selector. If empty or not present, the service is assumed to have an
	// external process managing its endpoints, which Kubernetes will not
	// modify. Only applies to types ClusterIP, NodePort, and LoadBalancer.
	// Ignored if type is ExternalName.
	// More info: https://kubernetes.io/docs/concepts/services-networking/service/
	Selector map[string]string

	// ClusterIP is the IP address of the service and is usually assigned
	// randomly by the master. If an address is specified manually and is not in
	// use by others, it will be allocated to the service; otherwise, creation
	// of the service will fail. This field can not be changed through updates.
	// Valid values are "None", empty string (""), or a valid IP address. "None"
	// can be specified for headless services when proxying is not required.
	// Only applies to types ClusterIP, NodePort, and LoadBalancer. Ignored if
	// type is ExternalName.
	// More info: https://kubernetes.io/docs/concepts/services-networking/service/#virtual-ips-and-service-proxies
	// +optional
	ClusterIP string

	// ExternalName is the external reference that kubedns or equivalent will
	// return as a CNAME record for this service. No proxying will be involved.
	// Must be a valid RFC-1123 hostname (https://tools.ietf.org/html/rfc1123)
	// and requires Type to be ExternalName.
	ExternalName string

	// ExternalIPs are used by external load balancers, or can be set by
	// users to handle external traffic that arrives at a node.
	// +optional
	ExternalIPs []string

	// Only applies to Service Type: LoadBalancer
	// LoadBalancer will get created with the IP specified in this field.
	// This feature depends on whether the underlying cloud-provider supports specifying
	// the loadBalancerIP when a load balancer is created.
	// This field will be ignored if the cloud-provider does not support the feature.
	// +optional
	LoadBalancerIP string

	// Optional: Supports "ClientIP" and "None".  Used to maintain session affinity.
	// +optional
	SessionAffinity ServiceAffinity

	// sessionAffinityConfig contains the configurations of session affinity.
	// +optional
	SessionAffinityConfig *SessionAffinityConfig

	// Optional: If specified and supported by the platform, this will restrict traffic through the cloud-provider
	// load-balancer will be restricted to the specified client IPs. This field will be ignored if the
	// cloud-provider does not support the feature."
	// +optional
	LoadBalancerSourceRanges []string

	// externalTrafficPolicy denotes if this Service desires to route external
	// traffic to node-local or cluster-wide endpoints. "Local" preserves the
	// client source IP and avoids a second hop for LoadBalancer and Nodeport
	// type services, but risks potentially imbalanced traffic spreading.
	// "Cluster" obscures the client source IP and may cause a second hop to
	// another node, but should have good overall load-spreading.
	// +optional
	ExternalTrafficPolicy ServiceExternalTrafficPolicyType

	// healthCheckNodePort specifies the healthcheck nodePort for the service.
	// If not specified, HealthCheckNodePort is created by the service api
	// backend with the allocated nodePort. Will use user-specified nodePort value
	// if specified by the client. Only effects when Type is set to LoadBalancer
	// and ExternalTrafficPolicy is set to Local.
	// +optional
	HealthCheckNodePort int32

	// publishNotReadyAddresses, when set to true, indicates that DNS implementations
	// must publish the notReadyAddresses of subsets for the Endpoints associated with
	// the Service. The default value is false.
	// The primary use case for setting this field is to use a StatefulSet's Headless Service
	// to propagate SRV records for its Pods without respect to their readiness for purpose
	// of peer discovery.
	// +optional
	PublishNotReadyAddresses bool
}

type ServicePort struct {
	// Optional if only one ServicePort is defined on this service: The
	// name of this port within the service.  This must be a DNS_LABEL.
	// All ports within a ServiceSpec must have unique names.  This maps to
	// the 'Name' field in EndpointPort objects.
	Name string

	// The IP protocol for this port.  Supports "TCP", "UDP", and "SCTP".
	Protocol Protocol

	// The port that will be exposed on the service.
	Port int32

	// Optional: The target port on pods selected by this service.  If this
	// is a string, it will be looked up as a named port in the target
	// Pod's container ports.  If this is not specified, the value
	// of the 'port' field is used (an identity map).
	// This field is ignored for services with clusterIP=None, and should be
	// omitted or set equal to the 'port' field.
	TargetPort intstr.IntOrString

	// The port on each node on which this service is exposed.
	// Default is to auto-allocate a port if the ServiceType of this Service requires one.
	NodePort int32
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Service is a named abstraction of software service (for example, mysql) consisting of local port
// (for example 3306) that the proxy listens on, and the selector that determines which pods
// will answer requests sent through the proxy.
type Service struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the behavior of a service.
	// +optional
	Spec ServiceSpec

	// Status represents the current status of a service.
	// +optional
	Status ServiceStatus
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceAccount binds together:
// * a name, understood by users, and perhaps by peripheral systems, for an identity
// * a principal that can be authenticated and authorized
// * a set of secrets
type ServiceAccount struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Secrets is the list of secrets allowed to be used by pods running using this ServiceAccount
	Secrets []ObjectReference

	// ImagePullSecrets is a list of references to secrets in the same namespace to use for pulling any images
	// in pods that reference this ServiceAccount.  ImagePullSecrets are distinct from Secrets because Secrets
	// can be mounted in the pod, but ImagePullSecrets are only accessed by the kubelet.
	// +optional
	ImagePullSecrets []LocalObjectReference

	// AutomountServiceAccountToken indicates whether pods running as this service account should have an API token automatically mounted.
	// Can be overridden at the pod level.
	// +optional
	AutomountServiceAccountToken *bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceAccountList is a list of ServiceAccount objects
type ServiceAccountList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []ServiceAccount
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Endpoints is a collection of endpoints that implement the actual service.  Example:
//   Name: "mysvc",
//   Subsets: [
//     {
//       Addresses: [{"ip": "10.10.1.1"}, {"ip": "10.10.2.2"}],
//       Ports: [{"name": "a", "port": 8675}, {"name": "b", "port": 309}]
//     },
//     {
//       Addresses: [{"ip": "10.10.3.3"}],
//       Ports: [{"name": "a", "port": 93}, {"name": "b", "port": 76}]
//     },
//  ]
type Endpoints struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// The set of all endpoints is the union of all subsets.
	Subsets []EndpointSubset
}

// EndpointSubset is a group of addresses with a common set of ports.  The
// expanded set of endpoints is the Cartesian product of Addresses x Ports.
// For example, given:
//   {
//     Addresses: [{"ip": "10.10.1.1"}, {"ip": "10.10.2.2"}],
//     Ports:     [{"name": "a", "port": 8675}, {"name": "b", "port": 309}]
//   }
// The resulting set of endpoints can be viewed as:
//     a: [ 10.10.1.1:8675, 10.10.2.2:8675 ],
//     b: [ 10.10.1.1:309, 10.10.2.2:309 ]
type EndpointSubset struct {
	Addresses         []EndpointAddress
	NotReadyAddresses []EndpointAddress
	Ports             []EndpointPort
}

// EndpointAddress is a tuple that describes single IP address.
type EndpointAddress struct {
	// The IP of this endpoint.
	// IPv6 is also accepted but not fully supported on all platforms. Also, certain
	// kubernetes components, like kube-proxy, are not IPv6 ready.
	// TODO: This should allow hostname or IP, see #4447.
	IP string
	// Optional: Hostname of this endpoint
	// Meant to be used by DNS servers etc.
	// +optional
	Hostname string
	// Optional: Node hosting this endpoint. This can be used to determine endpoints local to a node.
	// +optional
	NodeName *string
	// Optional: The kubernetes object related to the entry point.
	TargetRef *ObjectReference
}

// EndpointPort is a tuple that describes a single port.
type EndpointPort struct {
	// The name of this port (corresponds to ServicePort.Name).  Optional
	// if only one port is defined.  Must be a DNS_LABEL.
	Name string

	// The port number.
	Port int32

	// The IP protocol for this port.
	Protocol Protocol
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EndpointsList is a list of endpoints.
type EndpointsList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Endpoints
}

// NodeSpec describes the attributes that a node is created with.
type NodeSpec struct {
	// PodCIDR represents the pod IP range assigned to the node
	// Note: assigning IP ranges to nodes might need to be revisited when we support migratable IPs.
	// +optional
	PodCIDR string

	// ID of the node assigned by the cloud provider
	// Note: format is "<ProviderName>://<ProviderSpecificNodeID>"
	// +optional
	ProviderID string

	// Unschedulable controls node schedulability of new pods. By default node is schedulable.
	// +optional
	Unschedulable bool

	// If specified, the node's taints.
	// +optional
	Taints []Taint

	// If specified, the source to get node configuration from
	// The DynamicKubeletConfig feature gate must be enabled for the Kubelet to use this field
	// +optional
	ConfigSource *NodeConfigSource

	// Deprecated. Not all kubelets will set this field. Remove field after 1.13.
	// see: https://issues.k8s.io/61966
	// +optional
	DoNotUse_ExternalID string
}

// NodeConfigSource specifies a source of node configuration. Exactly one subfield must be non-nil.
type NodeConfigSource struct {
	ConfigMap *ConfigMapNodeConfigSource
}

type ConfigMapNodeConfigSource struct {
	// Namespace is the metadata.namespace of the referenced ConfigMap.
	// This field is required in all cases.
	Namespace string

	// Name is the metadata.name of the referenced ConfigMap.
	// This field is required in all cases.
	Name string

	// UID is the metadata.UID of the referenced ConfigMap.
	// This field is forbidden in Node.Spec, and required in Node.Status.
	// +optional
	UID types.UID

	// ResourceVersion is the metadata.ResourceVersion of the referenced ConfigMap.
	// This field is forbidden in Node.Spec, and required in Node.Status.
	// +optional
	ResourceVersion string

	// KubeletConfigKey declares which key of the referenced ConfigMap corresponds to the KubeletConfiguration structure
	// This field is required in all cases.
	KubeletConfigKey string
}

// DaemonEndpoint contains information about a single Daemon endpoint.
type DaemonEndpoint struct {
	/*
		The port tag was not properly in quotes in earlier releases, so it must be
		uppercase for backwards compatibility (since it was falling back to var name of
		'Port').
	*/

	// Port number of the given endpoint.
	Port int32
}

// NodeDaemonEndpoints lists ports opened by daemons running on the Node.
type NodeDaemonEndpoints struct {
	// Endpoint on which Kubelet is listening.
	// +optional
	KubeletEndpoint DaemonEndpoint
}

// NodeSystemInfo is a set of ids/uuids to uniquely identify the node.
type NodeSystemInfo struct {
	// MachineID reported by the node. For unique machine identification
	// in the cluster this field is preferred. Learn more from man(5)
	// machine-id: http://man7.org/linux/man-pages/man5/machine-id.5.html
	MachineID string
	// SystemUUID reported by the node. For unique machine identification
	// MachineID is preferred. This field is specific to Red Hat hosts
	// https://access.redhat.com/documentation/en-US/Red_Hat_Subscription_Management/1/html/RHSM/getting-system-uuid.html
	SystemUUID string
	// Boot ID reported by the node.
	BootID string
	// Kernel Version reported by the node.
	KernelVersion string
	// OS Image reported by the node.
	OSImage string
	// ContainerRuntime Version reported by the node.
	ContainerRuntimeVersion string
	// Kubelet Version reported by the node.
	KubeletVersion string
	// KubeProxy Version reported by the node.
	KubeProxyVersion string
	// The Operating System reported by the node
	OperatingSystem string
	// The Architecture reported by the node
	Architecture string
}

// NodeConfigStatus describes the status of the config assigned by Node.Spec.ConfigSource.
type NodeConfigStatus struct {
	// Assigned reports the checkpointed config the node will try to use.
	// When Node.Spec.ConfigSource is updated, the node checkpoints the associated
	// config payload to local disk, along with a record indicating intended
	// config. The node refers to this record to choose its config checkpoint, and
	// reports this record in Assigned. Assigned only updates in the status after
	// the record has been checkpointed to disk. When the Kubelet is restarted,
	// it tries to make the Assigned config the Active config by loading and
	// validating the checkpointed payload identified by Assigned.
	// +optional
	Assigned *NodeConfigSource
	// Active reports the checkpointed config the node is actively using.
	// Active will represent either the current version of the Assigned config,
	// or the current LastKnownGood config, depending on whether attempting to use the
	// Assigned config results in an error.
	// +optional
	Active *NodeConfigSource
	// LastKnownGood reports the checkpointed config the node will fall back to
	// when it encounters an error attempting to use the Assigned config.
	// The Assigned config becomes the LastKnownGood config when the node determines
	// that the Assigned config is stable and correct.
	// This is currently implemented as a 10-minute soak period starting when the local
	// record of Assigned config is updated. If the Assigned config is Active at the end
	// of this period, it becomes the LastKnownGood. Note that if Spec.ConfigSource is
	// reset to nil (use local defaults), the LastKnownGood is also immediately reset to nil,
	// because the local default config is always assumed good.
	// You should not make assumptions about the node's method of determining config stability
	// and correctness, as this may change or become configurable in the future.
	// +optional
	LastKnownGood *NodeConfigSource
	// Error describes any problems reconciling the Spec.ConfigSource to the Active config.
	// Errors may occur, for example, attempting to checkpoint Spec.ConfigSource to the local Assigned
	// record, attempting to checkpoint the payload associated with Spec.ConfigSource, attempting
	// to load or validate the Assigned config, etc.
	// Errors may occur at different points while syncing config. Earlier errors (e.g. download or
	// checkpointing errors) will not result in a rollback to LastKnownGood, and may resolve across
	// Kubelet retries. Later errors (e.g. loading or validating a checkpointed config) will result in
	// a rollback to LastKnownGood. In the latter case, it is usually possible to resolve the error
	// by fixing the config assigned in Spec.ConfigSource.
	// You can find additional information for debugging by searching the error message in the Kubelet log.
	// Error is a human-readable description of the error state; machines can check whether or not Error
	// is empty, but should not rely on the stability of the Error text across Kubelet versions.
	// +optional
	Error string
}

// NodeStatus is information about the current status of a node.
type NodeStatus struct {
	// Capacity represents the total resources of a node.
	// +optional
	Capacity ResourceList
	// Allocatable represents the resources of a node that are available for scheduling.
	// +optional
	Allocatable ResourceList
	// NodePhase is the current lifecycle phase of the node.
	// +optional
	Phase NodePhase
	// Conditions is an array of current node conditions.
	// +optional
	Conditions []NodeCondition
	// Queried from cloud provider, if available.
	// +optional
	Addresses []NodeAddress
	// Endpoints of daemons running on the Node.
	// +optional
	DaemonEndpoints NodeDaemonEndpoints
	// Set of ids/uuids to uniquely identify the node.
	// +optional
	NodeInfo NodeSystemInfo
	// List of container images on this node
	// +optional
	Images []ContainerImage
	// List of attachable volumes in use (mounted) by the node.
	// +optional
	VolumesInUse []UniqueVolumeName
	// List of volumes that are attached to the node.
	// +optional
	VolumesAttached []AttachedVolume
	// Status of the config assigned to the node via the dynamic Kubelet config feature.
	// +optional
	Config *NodeConfigStatus
}

type UniqueVolumeName string

// AttachedVolume describes a volume attached to a node
type AttachedVolume struct {
	// Name of the attached volume
	Name UniqueVolumeName

	// DevicePath represents the device path where the volume should be available
	DevicePath string
}

// AvoidPods describes pods that should avoid this node. This is the value for a
// Node annotation with key scheduler.alpha.kubernetes.io/preferAvoidPods and
// will eventually become a field of NodeStatus.
type AvoidPods struct {
	// Bounded-sized list of signatures of pods that should avoid this node, sorted
	// in timestamp order from oldest to newest. Size of the slice is unspecified.
	// +optional
	PreferAvoidPods []PreferAvoidPodsEntry
}

// Describes a class of pods that should avoid this node.
type PreferAvoidPodsEntry struct {
	// The class of pods.
	PodSignature PodSignature
	// Time at which this entry was added to the list.
	// +optional
	EvictionTime metav1.Time
	// (brief) reason why this entry was added to the list.
	// +optional
	Reason string
	// Human readable message indicating why this entry was added to the list.
	// +optional
	Message string
}

// Describes the class of pods that should avoid this node.
// Exactly one field should be set.
type PodSignature struct {
	// Reference to controller whose pods should avoid this node.
	// +optional
	PodController *metav1.OwnerReference
}

// Describe a container image
type ContainerImage struct {
	// Names by which this image is known.
	Names []string
	// The size of the image in bytes.
	// +optional
	SizeBytes int64
}

type NodePhase string

// These are the valid phases of node.
const (
	// NodePending means the node has been created/added by the system, but not configured.
	NodePending NodePhase = "Pending"
	// NodeRunning means the node has been configured and has Kubernetes components running.
	NodeRunning NodePhase = "Running"
	// NodeTerminated means the node has been removed from the cluster.
	NodeTerminated NodePhase = "Terminated"
)

type NodeConditionType string

// These are valid conditions of node. Currently, we don't have enough information to decide
// node condition. In the future, we will add more. The proposed set of conditions are:
// NodeReady, NodeReachable
const (
	// NodeReady means kubelet is healthy and ready to accept pods.
	NodeReady NodeConditionType = "Ready"
	// NodeOutOfDisk means the kubelet will not accept new pods due to insufficient free disk
	// space on the node.
	NodeOutOfDisk NodeConditionType = "OutOfDisk"
	// NodeMemoryPressure means the kubelet is under pressure due to insufficient available memory.
	NodeMemoryPressure NodeConditionType = "MemoryPressure"
	// NodeDiskPressure means the kubelet is under pressure due to insufficient available disk.
	NodeDiskPressure NodeConditionType = "DiskPressure"
	// NodeNetworkUnavailable means that network for the node is not correctly configured.
	NodeNetworkUnavailable NodeConditionType = "NetworkUnavailable"
)

type NodeCondition struct {
	Type   NodeConditionType
	Status ConditionStatus
	// +optional
	LastHeartbeatTime metav1.Time
	// +optional
	LastTransitionTime metav1.Time
	// +optional
	Reason string
	// +optional
	Message string
}

type NodeAddressType string

const (
	NodeHostName    NodeAddressType = "Hostname"
	NodeExternalIP  NodeAddressType = "ExternalIP"
	NodeInternalIP  NodeAddressType = "InternalIP"
	NodeExternalDNS NodeAddressType = "ExternalDNS"
	NodeInternalDNS NodeAddressType = "InternalDNS"
)

type NodeAddress struct {
	Type    NodeAddressType
	Address string
}

// NodeResources is an object for conveying resource information about a node.
// see http://releases.k8s.io/HEAD/docs/design/resources.md for more details.
type NodeResources struct {
	// Capacity represents the available resources of a node
	// +optional
	Capacity ResourceList
}

// ResourceName is the name identifying various resources in a ResourceList.
type ResourceName string

// Resource names must be not more than 63 characters, consisting of upper- or lower-case alphanumeric characters,
// with the -, _, and . characters allowed anywhere, except the first or last character.
// The default convention, matching that for annotations, is to use lower-case names, with dashes, rather than
// camel case, separating compound words.
// Fully-qualified resource typenames are constructed from a DNS-style subdomain, followed by a slash `/` and a name.
const (
	// CPU, in cores. (500m = .5 cores)
	ResourceCPU ResourceName = "cpu"
	// Memory, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceMemory ResourceName = "memory"
	// Volume size, in bytes (e,g. 5Gi = 5GiB = 5 * 1024 * 1024 * 1024)
	ResourceStorage ResourceName = "storage"
	// Local ephemeral storage, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	// The resource name for ResourceEphemeralStorage is alpha and it can change across releases.
	ResourceEphemeralStorage ResourceName = "ephemeral-storage"
)

const (
	// Default namespace prefix.
	ResourceDefaultNamespacePrefix = "kubernetes.io/"
	// Name prefix for huge page resources (alpha).
	ResourceHugePagesPrefix = "hugepages-"
	// Name prefix for storage resource limits
	ResourceAttachableVolumesPrefix = "attachable-volumes-"
)

// ResourceList is a set of (resource name, quantity) pairs.
type ResourceList map[ResourceName]resource.Quantity

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Node is a worker node in Kubernetes
// The name of the node according to etcd is in ObjectMeta.Name.
type Node struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the behavior of a node.
	// +optional
	Spec NodeSpec

	// Status describes the current status of a Node
	// +optional
	Status NodeStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeList is a list of nodes.
type NodeList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Node
}

// NamespaceSpec describes the attributes on a Namespace
type NamespaceSpec struct {
	// Finalizers is an opaque list of values that must be empty to permanently remove object from storage
	Finalizers []FinalizerName
}

// FinalizerName is the name identifying a finalizer during namespace lifecycle.
type FinalizerName string

// These are internal finalizer values to Kubernetes, must be qualified name unless defined here or
// in metav1.
const (
	FinalizerKubernetes FinalizerName = "kubernetes"
)

// NamespaceStatus is information about the current status of a Namespace.
type NamespaceStatus struct {
	// Phase is the current lifecycle phase of the namespace.
	// +optional
	Phase NamespacePhase
}

type NamespacePhase string

// These are the valid phases of a namespace.
const (
	// NamespaceActive means the namespace is available for use in the system
	NamespaceActive NamespacePhase = "Active"
	// NamespaceTerminating means the namespace is undergoing graceful termination
	NamespaceTerminating NamespacePhase = "Terminating"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// A namespace provides a scope for Names.
// Use of multiple namespaces is optional
type Namespace struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the behavior of the Namespace.
	// +optional
	Spec NamespaceSpec

	// Status describes the current status of a Namespace
	// +optional
	Status NamespaceStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NamespaceList is a list of Namespaces.
type NamespaceList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Namespace
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Binding ties one object to another; for example, a pod is bound to a node by a scheduler.
// Deprecated in 1.7, please use the bindings subresource of pods instead.
type Binding struct {
	metav1.TypeMeta
	// ObjectMeta describes the object that is being bound.
	// +optional
	metav1.ObjectMeta

	// Target is the object to bind to.
	Target ObjectReference
}

// Preconditions must be fulfilled before an operation (update, delete, etc.) is carried out.
type Preconditions struct {
	// Specifies the target UID.
	// +optional
	UID *types.UID
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodLogOptions is the query options for a Pod's logs REST call
type PodLogOptions struct {
	metav1.TypeMeta

	// Container for which to return logs
	Container string
	// If true, follow the logs for the pod
	Follow bool
	// If true, return previous terminated container logs
	Previous bool
	// A relative time in seconds before the current time from which to show logs. If this value
	// precedes the time a pod was started, only logs since the pod start will be returned.
	// If this value is in the future, no logs will be returned.
	// Only one of sinceSeconds or sinceTime may be specified.
	SinceSeconds *int64
	// An RFC3339 timestamp from which to show logs. If this value
	// precedes the time a pod was started, only logs since the pod start will be returned.
	// If this value is in the future, no logs will be returned.
	// Only one of sinceSeconds or sinceTime may be specified.
	SinceTime *metav1.Time
	// If true, add an RFC3339 or RFC3339Nano timestamp at the beginning of every line
	// of log output.
	Timestamps bool
	// If set, the number of lines from the end of the logs to show. If not specified,
	// logs are shown from the creation of the container or sinceSeconds or sinceTime
	TailLines *int64
	// If set, the number of bytes to read from the server before terminating the
	// log output. This may not display a complete final line of logging, and may return
	// slightly more or slightly less than the specified limit.
	LimitBytes *int64
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodAttachOptions is the query options to a Pod's remote attach call
// TODO: merge w/ PodExecOptions below for stdin, stdout, etc
type PodAttachOptions struct {
	metav1.TypeMeta

	// Stdin if true indicates that stdin is to be redirected for the attach call
	// +optional
	Stdin bool

	// Stdout if true indicates that stdout is to be redirected for the attach call
	// +optional
	Stdout bool

	// Stderr if true indicates that stderr is to be redirected for the attach call
	// +optional
	Stderr bool

	// TTY if true indicates that a tty will be allocated for the attach call
	// +optional
	TTY bool

	// Container to attach to.
	// +optional
	Container string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodExecOptions is the query options to a Pod's remote exec call
type PodExecOptions struct {
	metav1.TypeMeta

	// Stdin if true indicates that stdin is to be redirected for the exec call
	Stdin bool

	// Stdout if true indicates that stdout is to be redirected for the exec call
	Stdout bool

	// Stderr if true indicates that stderr is to be redirected for the exec call
	Stderr bool

	// TTY if true indicates that a tty will be allocated for the exec call
	TTY bool

	// Container in which to execute the command.
	Container string

	// Command is the remote command to execute; argv array; not executed within a shell.
	Command []string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodPortForwardOptions is the query options to a Pod's port forward call
type PodPortForwardOptions struct {
	metav1.TypeMeta

	// The list of ports to forward
	// +optional
	Ports []int32
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodProxyOptions is the query options to a Pod's proxy call
type PodProxyOptions struct {
	metav1.TypeMeta

	// Path is the URL path to use for the current proxy request
	Path string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeProxyOptions is the query options to a Node's proxy call
type NodeProxyOptions struct {
	metav1.TypeMeta

	// Path is the URL path to use for the current proxy request
	Path string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceProxyOptions is the query options to a Service's proxy call.
type ServiceProxyOptions struct {
	metav1.TypeMeta

	// Path is the part of URLs that include service endpoints, suffixes,
	// and parameters to use for the current proxy request to service.
	// For example, the whole request URL is
	// http://localhost/api/v1/namespaces/kube-system/services/elasticsearch-logging/_search?q=user:kimchy.
	// Path is _search?q=user:kimchy.
	Path string
}

// ObjectReference contains enough information to let you inspect or modify the referred object.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ObjectReference struct {
	// +optional
	Kind string
	// +optional
	Namespace string
	// +optional
	Name string
	// +optional
	UID types.UID
	// +optional
	APIVersion string
	// +optional
	ResourceVersion string

	// Optional. If referring to a piece of an object instead of an entire object, this string
	// should contain information to identify the sub-object. For example, if the object
	// reference is to a container within a pod, this would take on a value like:
	// "spec.containers{name}" (where "name" refers to the name of the container that triggered
	// the event) or if no container name is specified "spec.containers[2]" (container with
	// index 2 in this pod). This syntax is chosen only to have some well-defined way of
	// referencing a part of an object.
	// TODO: this design is not final and this field is subject to change in the future.
	// +optional
	FieldPath string
}

// LocalObjectReference contains enough information to let you locate the referenced object inside the same namespace.
type LocalObjectReference struct {
	//TODO: Add other useful fields.  apiVersion, kind, uid?
	Name string
}

// TypedLocalObjectReference contains enough information to let you locate the typed referenced object inside the same namespace.
type TypedLocalObjectReference struct {
	// APIGroup is the group for the resource being referenced.
	// If APIGroup is not specified, the specified Kind must be in the core API group.
	// For any other third-party types, APIGroup is required.
	// +optional
	APIGroup *string
	// Kind is the type of resource being referenced
	Kind string
	// Name is the name of resource being referenced
	Name string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type SerializedReference struct {
	metav1.TypeMeta
	// +optional
	Reference ObjectReference
}

type EventSource struct {
	// Component from which the event is generated.
	// +optional
	Component string
	// Node name on which the event is generated.
	// +optional
	Host string
}

// Valid values for event types (new types could be added in future)
const (
	// Information only and will not cause any problems
	EventTypeNormal string = "Normal"
	// These events are to warn that something might go wrong
	EventTypeWarning string = "Warning"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Event is a report of an event somewhere in the cluster.
// TODO: Decide whether to store these separately or with the object they apply to.
type Event struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Required. The object that this event is about. Mapped to events.Event.regarding
	// +optional
	InvolvedObject ObjectReference

	// Optional; this should be a short, machine understandable string that gives the reason
	// for this event being generated. For example, if the event is reporting that a container
	// can't start, the Reason might be "ImageNotFound".
	// TODO: provide exact specification for format.
	// +optional
	Reason string

	// Optional. A human-readable description of the status of this operation.
	// TODO: decide on maximum length. Mapped to events.Event.note
	// +optional
	Message string

	// Optional. The component reporting this event. Should be a short machine understandable string.
	// +optional
	Source EventSource

	// The time at which the event was first recorded. (Time of server receipt is in TypeMeta.)
	// +optional
	FirstTimestamp metav1.Time

	// The time at which the most recent occurrence of this event was recorded.
	// +optional
	LastTimestamp metav1.Time

	// The number of times this event has occurred.
	// +optional
	Count int32

	// Type of this event (Normal, Warning), new types could be added in the future.
	// +optional
	Type string

	// Time when this Event was first observed.
	// +optional
	EventTime metav1.MicroTime

	// Data about the Event series this event represents or nil if it's a singleton Event.
	// +optional
	Series *EventSeries

	// What action was taken/failed regarding to the Regarding object.
	// +optional
	Action string

	// Optional secondary object for more complex actions.
	// +optional
	Related *ObjectReference

	// Name of the controller that emitted this Event, e.g. `kubernetes.io/kubelet`.
	// +optional
	ReportingController string

	// ID of the controller instance, e.g. `kubelet-xyzf`.
	// +optional
	ReportingInstance string
}

type EventSeries struct {
	// Number of occurrences in this series up to the last heartbeat time
	Count int32
	// Time of the last occurrence observed
	LastObservedTime metav1.MicroTime
	// State of this Series: Ongoing or Finished
	State EventSeriesState
}

type EventSeriesState string

const (
	EventSeriesStateOngoing  EventSeriesState = "Ongoing"
	EventSeriesStateFinished EventSeriesState = "Finished"
	EventSeriesStateUnknown  EventSeriesState = "Unknown"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EventList is a list of events.
type EventList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Event
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// List holds a list of objects, which may not be known by the server.
type List metainternalversion.List

// A type of object that is limited
type LimitType string

const (
	// Limit that applies to all pods in a namespace
	LimitTypePod LimitType = "Pod"
	// Limit that applies to all containers in a namespace
	LimitTypeContainer LimitType = "Container"
	// Limit that applies to all persistent volume claims in a namespace
	LimitTypePersistentVolumeClaim LimitType = "PersistentVolumeClaim"
)

// LimitRangeItem defines a min/max usage limit for any resource that matches on kind
type LimitRangeItem struct {
	// Type of resource that this limit applies to
	// +optional
	Type LimitType
	// Max usage constraints on this kind by resource name
	// +optional
	Max ResourceList
	// Min usage constraints on this kind by resource name
	// +optional
	Min ResourceList
	// Default resource requirement limit value by resource name.
	// +optional
	Default ResourceList
	// DefaultRequest resource requirement request value by resource name.
	// +optional
	DefaultRequest ResourceList
	// MaxLimitRequestRatio represents the max burst value for the named resource
	// +optional
	MaxLimitRequestRatio ResourceList
}

// LimitRangeSpec defines a min/max usage limit for resources that match on kind
type LimitRangeSpec struct {
	// Limits is the list of LimitRangeItem objects that are enforced
	Limits []LimitRangeItem
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LimitRange sets resource usage limits for each kind of resource in a Namespace
type LimitRange struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the limits enforced
	// +optional
	Spec LimitRangeSpec
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LimitRangeList is a list of LimitRange items.
type LimitRangeList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is a list of LimitRange objects
	Items []LimitRange
}

// The following identify resource constants for Kubernetes object types
const (
	// Pods, number
	ResourcePods ResourceName = "pods"
	// Services, number
	ResourceServices ResourceName = "services"
	// ReplicationControllers, number
	ResourceReplicationControllers ResourceName = "replicationcontrollers"
	// ResourceQuotas, number
	ResourceQuotas ResourceName = "resourcequotas"
	// ResourceSecrets, number
	ResourceSecrets ResourceName = "secrets"
	// ResourceConfigMaps, number
	ResourceConfigMaps ResourceName = "configmaps"
	// ResourcePersistentVolumeClaims, number
	ResourcePersistentVolumeClaims ResourceName = "persistentvolumeclaims"
	// ResourceServicesNodePorts, number
	ResourceServicesNodePorts ResourceName = "services.nodeports"
	// ResourceServicesLoadBalancers, number
	ResourceServicesLoadBalancers ResourceName = "services.loadbalancers"
	// CPU request, in cores. (500m = .5 cores)
	ResourceRequestsCPU ResourceName = "requests.cpu"
	// Memory request, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceRequestsMemory ResourceName = "requests.memory"
	// Storage request, in bytes
	ResourceRequestsStorage ResourceName = "requests.storage"
	// Local ephemeral storage request, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceRequestsEphemeralStorage ResourceName = "requests.ephemeral-storage"
	// CPU limit, in cores. (500m = .5 cores)
	ResourceLimitsCPU ResourceName = "limits.cpu"
	// Memory limit, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceLimitsMemory ResourceName = "limits.memory"
	// Local ephemeral storage limit, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceLimitsEphemeralStorage ResourceName = "limits.ephemeral-storage"
)

// The following identify resource prefix for Kubernetes object types
const (
	// HugePages request, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	// As burst is not supported for HugePages, we would only quota its request, and ignore the limit.
	ResourceRequestsHugePagesPrefix = "requests.hugepages-"
	// Default resource requests prefix
	DefaultResourceRequestsPrefix = "requests."
)

// A ResourceQuotaScope defines a filter that must match each object tracked by a quota
type ResourceQuotaScope string

const (
	// Match all pod objects where spec.activeDeadlineSeconds
	ResourceQuotaScopeTerminating ResourceQuotaScope = "Terminating"
	// Match all pod objects where !spec.activeDeadlineSeconds
	ResourceQuotaScopeNotTerminating ResourceQuotaScope = "NotTerminating"
	// Match all pod objects that have best effort quality of service
	ResourceQuotaScopeBestEffort ResourceQuotaScope = "BestEffort"
	// Match all pod objects that do not have best effort quality of service
	ResourceQuotaScopeNotBestEffort ResourceQuotaScope = "NotBestEffort"
	// Match all pod objects that have priority class mentioned
	ResourceQuotaScopePriorityClass ResourceQuotaScope = "PriorityClass"
)

// ResourceQuotaSpec defines the desired hard limits to enforce for Quota
type ResourceQuotaSpec struct {
	// Hard is the set of desired hard limits for each named resource
	// +optional
	Hard ResourceList
	// A collection of filters that must match each object tracked by a quota.
	// If not specified, the quota matches all objects.
	// +optional
	Scopes []ResourceQuotaScope
	// ScopeSelector is also a collection of filters like Scopes that must match each object tracked by a quota
	// but expressed using ScopeSelectorOperator in combination with possible values.
	// +optional
	ScopeSelector *ScopeSelector
}

// A scope selector represents the AND of the selectors represented
// by the scoped-resource selector terms.
type ScopeSelector struct {
	// A list of scope selector requirements by scope of the resources.
	// +optional
	MatchExpressions []ScopedResourceSelectorRequirement
}

// A scoped-resource selector requirement is a selector that contains values, a scope name, and an operator
// that relates the scope name and values.
type ScopedResourceSelectorRequirement struct {
	// The name of the scope that the selector applies to.
	ScopeName ResourceQuotaScope
	// Represents a scope's relationship to a set of values.
	// Valid operators are In, NotIn, Exists, DoesNotExist.
	Operator ScopeSelectorOperator
	// An array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty.
	// This array is replaced during a strategic merge patch.
	// +optional
	Values []string
}

// A scope selector operator is the set of operators that can be used in
// a scope selector requirement.
type ScopeSelectorOperator string

const (
	ScopeSelectorOpIn           ScopeSelectorOperator = "In"
	ScopeSelectorOpNotIn        ScopeSelectorOperator = "NotIn"
	ScopeSelectorOpExists       ScopeSelectorOperator = "Exists"
	ScopeSelectorOpDoesNotExist ScopeSelectorOperator = "DoesNotExist"
)

// ResourceQuotaStatus defines the enforced hard limits and observed use
type ResourceQuotaStatus struct {
	// Hard is the set of enforced hard limits for each named resource
	// +optional
	Hard ResourceList
	// Used is the current observed total usage of the resource in the namespace
	// +optional
	Used ResourceList
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ResourceQuota sets aggregate quota restrictions enforced per namespace
type ResourceQuota struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the desired quota
	// +optional
	Spec ResourceQuotaSpec

	// Status defines the actual enforced quota and its current usage
	// +optional
	Status ResourceQuotaStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ResourceQuotaList is a list of ResourceQuota items
type ResourceQuotaList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is a list of ResourceQuota objects
	Items []ResourceQuota
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Secret holds secret data of a certain type.  The total bytes of the values in
// the Data field must be less than MaxSecretSize bytes.
type Secret struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Data contains the secret data. Each key must consist of alphanumeric
	// characters, '-', '_' or '.'. The serialized form of the secret data is a
	// base64 encoded string, representing the arbitrary (possibly non-string)
	// data value here.
	// +optional
	Data map[string][]byte

	// Used to facilitate programmatic handling of secret data.
	// +optional
	Type SecretType
}

const MaxSecretSize = 1 * 1024 * 1024

type SecretType string

const (
	// SecretTypeOpaque is the default; arbitrary user-defined data
	SecretTypeOpaque SecretType = "Opaque"

	// SecretTypeServiceAccountToken contains a token that identifies a service account to the API
	//
	// Required fields:
	// - Secret.Annotations["kubernetes.io/service-account.name"] - the name of the ServiceAccount the token identifies
	// - Secret.Annotations["kubernetes.io/service-account.uid"] - the UID of the ServiceAccount the token identifies
	// - Secret.Data["token"] - a token that identifies the service account to the API
	SecretTypeServiceAccountToken SecretType = "kubernetes.io/service-account-token"

	// ServiceAccountNameKey is the key of the required annotation for SecretTypeServiceAccountToken secrets
	ServiceAccountNameKey = "kubernetes.io/service-account.name"
	// ServiceAccountUIDKey is the key of the required annotation for SecretTypeServiceAccountToken secrets
	ServiceAccountUIDKey = "kubernetes.io/service-account.uid"
	// ServiceAccountTokenKey is the key of the required data for SecretTypeServiceAccountToken secrets
	ServiceAccountTokenKey = "token"
	// ServiceAccountKubeconfigKey is the key of the optional kubeconfig data for SecretTypeServiceAccountToken secrets
	ServiceAccountKubeconfigKey = "kubernetes.kubeconfig"
	// ServiceAccountRootCAKey is the key of the optional root certificate authority for SecretTypeServiceAccountToken secrets
	ServiceAccountRootCAKey = "ca.crt"
	// ServiceAccountNamespaceKey is the key of the optional namespace to use as the default for namespaced API calls
	ServiceAccountNamespaceKey = "namespace"

	// SecretTypeDockercfg contains a dockercfg file that follows the same format rules as ~/.dockercfg
	//
	// Required fields:
	// - Secret.Data[".dockercfg"] - a serialized ~/.dockercfg file
	SecretTypeDockercfg SecretType = "kubernetes.io/dockercfg"

	// DockerConfigKey is the key of the required data for SecretTypeDockercfg secrets
	DockerConfigKey = ".dockercfg"

	// SecretTypeDockerConfigJson contains a dockercfg file that follows the same format rules as ~/.docker/config.json
	//
	// Required fields:
	// - Secret.Data[".dockerconfigjson"] - a serialized ~/.docker/config.json file
	SecretTypeDockerConfigJson SecretType = "kubernetes.io/dockerconfigjson"

	// DockerConfigJsonKey is the key of the required data for SecretTypeDockerConfigJson secrets
	DockerConfigJsonKey = ".dockerconfigjson"

	// SecretTypeBasicAuth contains data needed for basic authentication.
	//
	// Required at least one of fields:
	// - Secret.Data["username"] - username used for authentication
	// - Secret.Data["password"] - password or token needed for authentication
	SecretTypeBasicAuth SecretType = "kubernetes.io/basic-auth"

	// BasicAuthUsernameKey is the key of the username for SecretTypeBasicAuth secrets
	BasicAuthUsernameKey = "username"
	// BasicAuthPasswordKey is the key of the password or token for SecretTypeBasicAuth secrets
	BasicAuthPasswordKey = "password"

	// SecretTypeSSHAuth contains data needed for SSH authentication.
	//
	// Required field:
	// - Secret.Data["ssh-privatekey"] - private SSH key needed for authentication
	SecretTypeSSHAuth SecretType = "kubernetes.io/ssh-auth"

	// SSHAuthPrivateKey is the key of the required SSH private key for SecretTypeSSHAuth secrets
	SSHAuthPrivateKey = "ssh-privatekey"

	// SecretTypeTLS contains information about a TLS client or server secret. It
	// is primarily used with TLS termination of the Ingress resource, but may be
	// used in other types.
	//
	// Required fields:
	// - Secret.Data["tls.key"] - TLS private key.
	//   Secret.Data["tls.crt"] - TLS certificate.
	// TODO: Consider supporting different formats, specifying CA/destinationCA.
	SecretTypeTLS SecretType = "kubernetes.io/tls"

	// TLSCertKey is the key for tls certificates in a TLS secret.
	TLSCertKey = "tls.crt"
	// TLSPrivateKeyKey is the key for the private key field in a TLS secret.
	TLSPrivateKeyKey = "tls.key"
	// SecretTypeBootstrapToken is used during the automated bootstrap process (first
	// implemented by kubeadm). It stores tokens that are used to sign well known
	// ConfigMaps. They are used for authn.
	SecretTypeBootstrapToken SecretType = "bootstrap.kubernetes.io/token"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type SecretList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Secret
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ConfigMap holds configuration data for components or applications to consume.
type ConfigMap struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Data contains the configuration data.
	// Each key must consist of alphanumeric characters, '-', '_' or '.'.
	// Values with non-UTF-8 byte sequences must use the BinaryData field.
	// The keys stored in Data must not overlap with the keys in
	// the BinaryData field, this is enforced during validation process.
	// +optional
	Data map[string]string

	// BinaryData contains the binary data.
	// Each key must consist of alphanumeric characters, '-', '_' or '.'.
	// BinaryData can contain byte sequences that are not in the UTF-8 range.
	// The keys stored in BinaryData must not overlap with the ones in
	// the Data field, this is enforced during validation process.
	// Using this field will require 1.10+ apiserver and
	// kubelet.
	// +optional
	BinaryData map[string][]byte
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ConfigMapList is a resource containing a list of ConfigMap objects.
type ConfigMapList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is the list of ConfigMaps.
	Items []ConfigMap
}

// These constants are for remote command execution and port forwarding and are
// used by both the client side and server side components.
//
// This is probably not the ideal place for them, but it didn't seem worth it
// to create pkg/exec and pkg/portforward just to contain a single file with
// constants in it.  Suggestions for more appropriate alternatives are
// definitely welcome!
const (
	// Enable stdin for remote command execution
	ExecStdinParam = "input"
	// Enable stdout for remote command execution
	ExecStdoutParam = "output"
	// Enable stderr for remote command execution
	ExecStderrParam = "error"
	// Enable TTY for remote command execution
	ExecTTYParam = "tty"
	// Command to run for remote command execution
	ExecCommandParam = "command"

	// Name of header that specifies stream type
	StreamType = "streamType"
	// Value for streamType header for stdin stream
	StreamTypeStdin = "stdin"
	// Value for streamType header for stdout stream
	StreamTypeStdout = "stdout"
	// Value for streamType header for stderr stream
	StreamTypeStderr = "stderr"
	// Value for streamType header for data stream
	StreamTypeData = "data"
	// Value for streamType header for error stream
	StreamTypeError = "error"
	// Value for streamType header for terminal resize stream
	StreamTypeResize = "resize"

	// Name of header that specifies the port being forwarded
	PortHeader = "port"
	// Name of header that specifies a request ID used to associate the error
	// and data streams for a single forwarded connection
	PortForwardRequestIDHeader = "requestID"
)

// Type and constants for component health validation.
type ComponentConditionType string

// These are the valid conditions for the component.
const (
	ComponentHealthy ComponentConditionType = "Healthy"
)

type ComponentCondition struct {
	Type   ComponentConditionType
	Status ConditionStatus
	// +optional
	Message string
	// +optional
	Error string
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ComponentStatus (and ComponentStatusList) holds the cluster validation info.
type ComponentStatus struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// +optional
	Conditions []ComponentCondition
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type ComponentStatusList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []ComponentStatus
}

// SecurityContext holds security configuration that will be applied to a container.
// Some fields are present in both SecurityContext and PodSecurityContext.  When both
// are set, the values in SecurityContext take precedence.
type SecurityContext struct {
	// The capabilities to add/drop when running containers.
	// Defaults to the default set of capabilities granted by the container runtime.
	// +optional
	Capabilities *Capabilities
	// Run container in privileged mode.
	// Processes in privileged containers are essentially equivalent to root on the host.
	// Defaults to false.
	// +optional
	Privileged *bool
	// The SELinux context to be applied to the container.
	// If unspecified, the container runtime will allocate a random SELinux context for each
	// container.  May also be set in PodSecurityContext.  If set in both SecurityContext and
	// PodSecurityContext, the value specified in SecurityContext takes precedence.
	// +optional
	SELinuxOptions *SELinuxOptions
	// The UID to run the entrypoint of the container process.
	// Defaults to user specified in image metadata if unspecified.
	// May also be set in PodSecurityContext.  If set in both SecurityContext and
	// PodSecurityContext, the value specified in SecurityContext takes precedence.
	// +optional
	RunAsUser *int64
	// The GID to run the entrypoint of the container process.
	// Uses runtime default if unset.
	// May also be set in PodSecurityContext.  If set in both SecurityContext and
	// PodSecurityContext, the value specified in SecurityContext takes precedence.
	// +optional
	RunAsGroup *int64
	// Indicates that the container must run as a non-root user.
	// If true, the Kubelet will validate the image at runtime to ensure that it
	// does not run as UID 0 (root) and fail to start the container if it does.
	// If unset or false, no such validation will be performed.
	// May also be set in PodSecurityContext.  If set in both SecurityContext and
	// PodSecurityContext, the value specified in SecurityContext takes precedence.
	// +optional
	RunAsNonRoot *bool
	// The read-only root filesystem allows you to restrict the locations that an application can write
	// files to, ensuring the persistent data can only be written to mounts.
	// +optional
	ReadOnlyRootFilesystem *bool
	// AllowPrivilegeEscalation controls whether a process can gain more
	// privileges than its parent process. This bool directly controls if
	// the no_new_privs flag will be set on the container process.
	// +optional
	AllowPrivilegeEscalation *bool
	// ProcMount denotes the type of proc mount to use for the containers.
	// The default is DefaultProcMount which uses the container runtime defaults for
	// readonly paths and masked paths.
	// +optional
	ProcMount *ProcMountType
}

type ProcMountType string

const (
	// DefaultProcMount uses the container runtime defaults for readonly and masked
	// paths for /proc.  Most container runtimes mask certain paths in /proc to avoid
	// accidental security exposure of special devices or information.
	DefaultProcMount ProcMountType = "Default"

	// UnmaskedProcMount bypasses the default masking behavior of the container
	// runtime and ensures the newly created /proc the container stays intact with
	// no modifications.
	UnmaskedProcMount ProcMountType = "Unmasked"
)

// SELinuxOptions are the labels to be applied to the container.
type SELinuxOptions struct {
	// SELinux user label
	// +optional
	User string
	// SELinux role label
	// +optional
	Role string
	// SELinux type label
	// +optional
	Type string
	// SELinux level label.
	// +optional
	Level string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RangeAllocation is an opaque API object (not exposed to end users) that can be persisted to record
// the global allocation state of the cluster. The schema of Range and Data generic, in that Range
// should be a string representation of the inputs to a range (for instance, for IP allocation it
// might be a CIDR) and Data is an opaque blob understood by an allocator which is typically a
// binary range.  Consumers should use annotations to record additional information (schema version,
// data encoding hints). A range allocation should *ALWAYS* be recreatable at any time by observation
// of the cluster, thus the object is less strongly typed than most.
type RangeAllocation struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta
	// A string representing a unique label for a range of resources, such as a CIDR "10.0.0.0/8" or
	// port range "10000-30000". Range is not strongly schema'd here. The Range is expected to define
	// a start and end unless there is an implicit end.
	Range string
	// A byte array representing the serialized state of a range allocation. Additional clarifiers on
	// the type or format of data should be represented with annotations. For IP allocations, this is
	// represented as a bit array starting at the base IP of the CIDR in Range, with each bit representing
	// a single allocated address (the fifth bit on CIDR 10.0.0.0/8 is 10.0.0.4).
	Data []byte
}

const (
	// "default-scheduler" is the name of default scheduler.
	DefaultSchedulerName = "default-scheduler"

	// RequiredDuringScheduling affinity is not symmetric, but there is an implicit PreferredDuringScheduling affinity rule
	// corresponding to every RequiredDuringScheduling affinity rule.
	// When the --hard-pod-affinity-weight scheduler flag is not specified,
	// DefaultHardPodAffinityWeight defines the weight of the implicit PreferredDuringScheduling affinity rule.
	DefaultHardPodAffinitySymmetricWeight int32 = 1
)
