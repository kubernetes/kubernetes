/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api

import (
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

// Common string formats
// ---------------------
// Many fields in this API have formatting requirements.  The commonly used
// formats are defined here.
//
// C_IDENTIFIER:  This is a string that conforms to the definition of an "identifier"
//     in the C language.  This is captured by the following regex:
//         [A-Za-z_][A-Za-z0-9_]*
//     This defines the format, but not the length restriction, which should be
//     specified at the definition of any field of this type.
//
// DNS_LABEL:  This is a string, no more than 63 characters long, that conforms
//     to the definition of a "label" in RFCs 1035 and 1123.  This is captured
//     by the following regex:
//         [a-z0-9]([-a-z0-9]*[a-z0-9])?
//
// DNS_SUBDOMAIN:  This is a string, no more than 253 characters long, that conforms
//      to the definition of a "subdomain" in RFCs 1035 and 1123.  This is captured
//      by the following regex:
//         [a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*
//     or more simply:
//         DNS_LABEL(\.DNS_LABEL)*
//
// IANA_SVC_NAME: This is a string, no more than 15 characters long, that
//      conforms to the definition of IANA service name in RFC 6335.
//      It must contains at least one letter [a-z] and it must contains only [a-z0-9-].
//      Hypens ('-') cannot be leading or trailing character of the string
//      and cannot be adjacent to other hyphens.

// TypeMeta describes an individual object in an API response or request
// with strings representing the type of the object and its API schema version.
// Structures that are versioned or persisted should inline TypeMeta.
type TypeMeta struct {
	// Kind is a string value representing the REST resource this object represents.
	// Servers may infer this from the endpoint the client submits requests to.
	Kind string `json:"kind,omitempty"`

	// APIVersion defines the versioned schema of this representation of an object.
	// Servers should convert recognized schemas to the latest internal value, and
	// may reject unrecognized values.
	APIVersion string `json:"apiVersion,omitempty"`
}

// ListMeta describes metadata that synthetic resources must have, including lists and
// various status objects. A resource may have only one of {ObjectMeta, ListMeta}.
type ListMeta struct {
	// SelfLink is a URL representing this object.
	SelfLink string `json:"selfLink,omitempty"`

	// An opaque value that represents the version of this response for use with optimistic
	// concurrency and change monitoring endpoints.  Clients must treat these values as opaque
	// and values may only be valid for a particular resource or set of resources. Only servers
	// will generate resource versions.
	ResourceVersion string `json:"resourceVersion,omitempty"`
}

// ObjectMeta is metadata that all persisted resources must have, which includes all objects
// users must create.
type ObjectMeta struct {
	// Name is unique within a namespace.  Name is required when creating resources, although
	// some resources may allow a client to request the generation of an appropriate name
	// automatically. Name is primarily intended for creation idempotence and configuration
	// definition.
	Name string `json:"name,omitempty"`

	// GenerateName indicates that the name should be made unique by the server prior to persisting
	// it. A non-empty value for the field indicates the name will be made unique (and the name
	// returned to the client will be different than the name passed). The value of this field will
	// be combined with a unique suffix on the server if the Name field has not been provided.
	// The provided value must be valid within the rules for Name, and may be truncated by the length
	// of the suffix required to make the value unique on the server.
	//
	// If this field is specified, and Name is not present, the server will NOT return a 409 if the
	// generated name exists - instead, it will either return 201 Created or 500 with Reason
	// ServerTimeout indicating a unique name could not be found in the time allotted, and the client
	// should retry (optionally after the time indicated in the Retry-After header).
	GenerateName string `json:"generateName,omitempty"`

	// Namespace defines the space within which name must be unique. An empty namespace is
	// equivalent to the "default" namespace, but "default" is the canonical representation.
	// Not all objects are required to be scoped to a namespace - the value of this field for
	// those objects will be empty.
	Namespace string `json:"namespace,omitempty"`

	// SelfLink is a URL representing this object.
	SelfLink string `json:"selfLink,omitempty"`

	// UID is the unique in time and space value for this object. It is typically generated by
	// the server on successful creation of a resource and is not allowed to change on PUT
	// operations.
	UID types.UID `json:"uid,omitempty"`

	// An opaque value that represents the version of this resource. May be used for optimistic
	// concurrency, change detection, and the watch operation on a resource or set of resources.
	// Clients must treat these values as opaque and values may only be valid for a particular
	// resource or set of resources. Only servers will generate resource versions.
	ResourceVersion string `json:"resourceVersion,omitempty"`

	// A sequence number representing a specific generation of the desired state.
	// Currently only implemented by replication controllers.
	Generation int64 `json:"generation,omitempty"`

	// CreationTimestamp is a timestamp representing the server time when this object was
	// created. It is not guaranteed to be set in happens-before order across separate operations.
	// Clients may not set this value. It is represented in RFC3339 form and is in UTC.
	CreationTimestamp util.Time `json:"creationTimestamp,omitempty"`

	// DeletionTimestamp is the time after which this resource will be deleted. This
	// field is set by the server when a graceful deletion is requested by the user, and is not
	// directly settable by a client. The resource will be deleted (no longer visible from
	// resource lists, and not reachable by name) after the time in this field. Once set, this
	// value may not be unset or be set further into the future, although it may be shortened
	// or the resource may be deleted prior to this time. For example, a user may request that
	// a pod is deleted in 30 seconds. The Kubelet will react by sending a graceful termination
	// signal to the containers in the pod. Once the resource is deleted in the API, the Kubelet
	// will send a hard termination signal to the container.
	DeletionTimestamp *util.Time `json:"deletionTimestamp,omitempty"`

	// Labels are key value pairs that may be used to scope and select individual resources.
	// Label keys are of the form:
	//     label-key ::= prefixed-name | name
	//     prefixed-name ::= prefix '/' name
	//     prefix ::= DNS_SUBDOMAIN
	//     name ::= DNS_LABEL
	// The prefix is optional.  If the prefix is not specified, the key is assumed to be private
	// to the user.  Other system components that wish to use labels must specify a prefix.  The
	// "kubernetes.io/" prefix is reserved for use by kubernetes components.
	// TODO: replace map[string]string with labels.LabelSet type
	Labels map[string]string `json:"labels,omitempty"`

	// Annotations are unstructured key value data stored with a resource that may be set by
	// external tooling. They are not queryable and should be preserved when modifying
	// objects.  Annotation keys have the same formatting restrictions as Label keys. See the
	// comments on Labels for details.
	Annotations map[string]string `json:"annotations,omitempty"`
}

const (
	// NamespaceDefault means the object is in the default namespace which is applied when not specified by clients
	NamespaceDefault string = "default"
	// NamespaceAll is the default argument to specify on a context when you want to list or filter resources across all namespaces
	NamespaceAll string = ""
	// NamespaceNone is the argument for a context when there is no namespace.
	NamespaceNone string = ""
	// NamespaceSystem is the system namespace where we place system components.
	NamespaceSystem string = "kube-system"
	// TerminationMessagePathDefault means the default path to capture the application termination message running in a container
	TerminationMessagePathDefault string = "/dev/termination-log"
)

// Volume represents a named volume in a pod that may be accessed by any containers in the pod.
type Volume struct {
	// Required: This must be a DNS_LABEL.  Each volume in a pod must have
	// a unique name.
	Name string `json:"name"`
	// The VolumeSource represents the location and type of a volume to mount.
	// This is optional for now. If not specified, the Volume is implied to be an EmptyDir.
	// This implied behavior is deprecated and will be removed in a future version.
	VolumeSource `json:",inline,omitempty"`
}

// VolumeSource represents the source location of a volume to mount.
// Only one of its members may be specified.
type VolumeSource struct {
	// HostPath represents file or directory on the host machine that is
	// directly exposed to the container. This is generally used for system
	// agents or other privileged things that are allowed to see the host
	// machine. Most containers will NOT need this.
	// TODO(jonesdl) We need to restrict who can use host directory mounts and who can/can not
	// mount host directories as read/write.
	HostPath *HostPathVolumeSource `json:"hostPath,omitempty"`
	// EmptyDir represents a temporary directory that shares a pod's lifetime.
	EmptyDir *EmptyDirVolumeSource `json:"emptyDir,omitempty"`
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDiskVolumeSource `json:"gcePersistentDisk,omitempty"`
	// AWSElasticBlockStore represents an AWS EBS disk that is attached to a
	// kubelet's host machine and then exposed to the pod.
	AWSElasticBlockStore *AWSElasticBlockStoreVolumeSource `json:"awsElasticBlockStore,omitempty"`
	// GitRepo represents a git repository at a particular revision.
	GitRepo *GitRepoVolumeSource `json:"gitRepo,omitempty"`
	// Secret represents a secret that should populate this volume.
	Secret *SecretVolumeSource `json:"secret,omitempty"`
	// NFS represents an NFS mount on the host that shares a pod's lifetime
	NFS *NFSVolumeSource `json:"nfs,omitempty"`
	// ISCSIVolumeSource represents an ISCSI Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	ISCSI *ISCSIVolumeSource `json:"iscsi,omitempty"`
	// Glusterfs represents a Glusterfs mount on the host that shares a pod's lifetime
	Glusterfs *GlusterfsVolumeSource `json:"glusterfs,omitempty"`
	// PersistentVolumeClaimVolumeSource represents a reference to a PersistentVolumeClaim in the same namespace
	PersistentVolumeClaim *PersistentVolumeClaimVolumeSource `json:"persistentVolumeClaim,omitempty"`
	// RBD represents a Rados Block Device mount on the host that shares a pod's lifetime
	RBD *RBDVolumeSource `json:"rbd,omitempty"`
}

// Similar to VolumeSource but meant for the administrator who creates PVs.
// Exactly one of its members must be set.
type PersistentVolumeSource struct {
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDiskVolumeSource `json:"gcePersistentDisk,omitempty"`
	// AWSElasticBlockStore represents an AWS EBS disk that is attached to a
	// kubelet's host machine and then exposed to the pod.
	AWSElasticBlockStore *AWSElasticBlockStoreVolumeSource `json:"awsElasticBlockStore,omitempty"`
	// HostPath represents a directory on the host.
	// This is useful for development and testing only.
	// on-host storage is not supported in any way
	HostPath *HostPathVolumeSource `json:"hostPath,omitempty"`
	// Glusterfs represents a Glusterfs volume that is attached to a host and exposed to the pod
	Glusterfs *GlusterfsVolumeSource `json:"glusterfs,omitempty"`
	// NFS represents an NFS mount on the host that shares a pod's lifetime
	NFS *NFSVolumeSource `json:"nfs,omitempty"`
	// RBD represents a Rados Block Device mount on the host that shares a pod's lifetime
	RBD *RBDVolumeSource `json:"rbd,omitempty"`
	// ISCSIVolumeSource represents an ISCSI resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	ISCSI *ISCSIVolumeSource `json:"iscsi,omitempty"`
}

type PersistentVolumeClaimVolumeSource struct {
	// ClaimName is the name of a PersistentVolumeClaim in the same namespace as the pod using this volume
	ClaimName string `json:"claimName"`
	// Optional: Defaults to false (read/write).  ReadOnly here
	// will force the ReadOnly setting in VolumeMounts
	ReadOnly bool `json:"readOnly,omitempty"`
}

type PersistentVolume struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	//Spec defines a persistent volume owned by the cluster
	Spec PersistentVolumeSpec `json:"spec,omitempty"`

	// Status represents the current information about persistent volume.
	Status PersistentVolumeStatus `json:"status,omitempty"`
}

type PersistentVolumeSpec struct {
	// Resources represents the actual resources of the volume
	Capacity ResourceList `json:"capacity"`
	// Source represents the location and type of a volume to mount.
	PersistentVolumeSource `json:",inline"`
	// AccessModes contains all ways the volume can be mounted
	AccessModes []PersistentVolumeAccessMode `json:"accessModes,omitempty"`
	// ClaimRef is part of a bi-directional binding between PersistentVolume and PersistentVolumeClaim.
	// ClaimRef is expected to be non-nil when bound.
	// claim.VolumeName is the authoritative bind between PV and PVC.
	ClaimRef *ObjectReference `json:"claimRef,omitempty"`
	// Optional: what happens to a persistent volume when released from its claim.
	PersistentVolumeReclaimPolicy PersistentVolumeReclaimPolicy `json:"persistentVolumeReclaimPolicy,omitempty"`
}

// PersistentVolumeReclaimPolicy describes a policy for end-of-life maintenance of persistent volumes
type PersistentVolumeReclaimPolicy string

const (
	// PersistentVolumeReclaimRecycle means the volume will be recycled back into the pool of unbound persistent volumes on release from its claim.
	// The volume plugin must support Recycling.
	PersistentVolumeReclaimRecycle PersistentVolumeReclaimPolicy = "Recycle"
	// PersistentVolumeReclaimDelete means the volume will be deleted from Kubernetes on release from its claim.
	// The volume plugin must support Deletion.
	// TODO: implement w/ DeletableVolumePlugin
	// PersistentVolumeReclaimDelete PersistentVolumeReclaimPolicy = "Delete"
	// PersistentVolumeReclaimRetain means the volume will left in its current phase (Released) for manual reclamation by the administrator.
	// The default policy is Retain.
	PersistentVolumeReclaimRetain PersistentVolumeReclaimPolicy = "Retain"
)

type PersistentVolumeStatus struct {
	// Phase indicates if a volume is available, bound to a claim, or released by a claim
	Phase PersistentVolumePhase `json:"phase,omitempty"`
	// A human-readable message indicating details about why the volume is in this state.
	Message string `json:"message,omitempty"`
	// Reason is a brief CamelCase string that describes any failure and is meant for machine parsing and tidy display in the CLI
	Reason string `json:"reason,omitempty"`
}

type PersistentVolumeList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`
	Items    []PersistentVolume `json:"items,omitempty"`
}

// PersistentVolumeClaim is a user's request for and claim to a persistent volume
type PersistentVolumeClaim struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the volume requested by a pod author
	Spec PersistentVolumeClaimSpec `json:"spec,omitempty"`

	// Status represents the current information about a claim
	Status PersistentVolumeClaimStatus `json:"status,omitempty"`
}

type PersistentVolumeClaimList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`
	Items    []PersistentVolumeClaim `json:"items,omitempty"`
}

// PersistentVolumeClaimSpec describes the common attributes of storage devices
// and allows a Source for provider-specific attributes
type PersistentVolumeClaimSpec struct {
	// Contains the types of access modes required
	AccessModes []PersistentVolumeAccessMode `json:"accessModes,omitempty"`
	// Resources represents the minimum resources required
	Resources ResourceRequirements `json:"resources,omitempty"`
	// VolumeName is the binding reference to the PersistentVolume backing this claim
	VolumeName string `json:"volumeName,omitempty"`
}

type PersistentVolumeClaimStatus struct {
	// Phase represents the current phase of PersistentVolumeClaim
	Phase PersistentVolumeClaimPhase `json:"phase,omitempty"`
	// AccessModes contains all ways the volume backing the PVC can be mounted
	AccessModes []PersistentVolumeAccessMode `json:"accessModes,omitempty"`
	// Represents the actual resources of the underlying volume
	Capacity ResourceList `json:"capacity,omitempty"`
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
)

// HostPathVolumeSource represents a host directory mapped into a pod.
type HostPathVolumeSource struct {
	Path string `json:"path"`
}

// EmptyDirVolumeSource represents an empty directory for a pod.
type EmptyDirVolumeSource struct {
	// TODO: Longer term we want to represent the selection of underlying
	// media more like a scheduling problem - user says what traits they
	// need, we give them a backing store that satisifies that.  For now
	// this will cover the most common needs.
	// Optional: what type of storage medium should back this directory.
	// The default is "" which means to use the node's default medium.
	Medium StorageMedium `json:"medium"`
}

// StorageMedium defines ways that storage can be allocated to a volume.
type StorageMedium string

const (
	StorageMediumDefault StorageMedium = ""       // use whatever the default is for the node
	StorageMediumMemory  StorageMedium = "Memory" // use memory (tmpfs)
)

// Protocol defines network protocols supported for things like conatiner ports.
type Protocol string

const (
	// ProtocolTCP is the TCP protocol.
	ProtocolTCP Protocol = "TCP"
	// ProtocolUDP is the UDP protocol.
	ProtocolUDP Protocol = "UDP"
)

// GCEPersistentDiskVolumeSource represents a Persistent Disk resource in Google Compute Engine.
//
// A GCE PD must exist and be formatted before mounting to a container.
// The disk must also be in the same GCE project and zone as the kubelet.
// A GCE PD can only be mounted as read/write once.
type GCEPersistentDiskVolumeSource struct {
	// Unique name of the PD resource. Used to identify the disk in GCE
	PDName string `json:"pdName"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType,omitempty"`
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field is 0 or empty.
	Partition int `json:"partition,omitempty"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty"`
}

// A ISCSI Disk can only be mounted as read/write once.
type ISCSIVolumeSource struct {
	// Required: iSCSI target portal
	// the portal is either an IP or ip_addr:port if port is other than default (typically TCP ports 860 and 3260)
	TargetPortal string `json:"targetPortal,omitempty"`
	// Required:  target iSCSI Qualified Name
	IQN string `json:"iqn,omitempty"`
	// Required: iSCSI target lun number
	Lun int `json:"lun,omitempty"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType,omitempty"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty"`
}

// AWSElasticBlockStoreVolumeSource represents a Persistent Disk resource in AWS.
//
// An AWS EBS disk must exist and be formatted before mounting to a container.
// The disk must also be in the same AWS zone as the kubelet.
// A AWS EBS disk can only be mounted as read/write once.
type AWSElasticBlockStoreVolumeSource struct {
	// Unique id of the persistent disk resource. Used to identify the disk in AWS
	VolumeID string `json:"volumeID"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType,omitempty"`
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field is 0 or empty.
	Partition int `json:"partition,omitempty"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty"`
}

// GitRepoVolumeSource represents a volume that is pulled from git when the pod is created.
type GitRepoVolumeSource struct {
	// Repository URL
	Repository string `json:"repository"`
	// Commit hash, this is optional
	Revision string `json:"revision"`
	// TODO: Consider credentials here.
}

// SecretVolumeSource adapts a Secret into a VolumeSource.
//
// The contents of the target Secret's Data field will be presented in a volume
// as files using the keys in the Data field as the file names.
type SecretVolumeSource struct {
	// Name of the secret in the pod's namespace to use
	SecretName string `json:"secretName"`
}

// NFSVolumeSource represents an NFS Mount that lasts the lifetime of a pod
type NFSVolumeSource struct {
	// Server is the hostname or IP address of the NFS server
	Server string `json:"server"`

	// Path is the exported NFS share
	Path string `json:"path"`

	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the NFS export to be mounted with read-only permissions
	ReadOnly bool `json:"readOnly,omitempty"`
}

// GlusterfsVolumeSource represents a Glusterfs Mount that lasts the lifetime of a pod
type GlusterfsVolumeSource struct {
	// Required: EndpointsName is the endpoint name that details Glusterfs topology
	EndpointsName string `json:"endpoints"`

	// Required: Path is the Glusterfs volume path
	Path string `json:"path"`

	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the Glusterfs to be mounted with read-only permissions
	ReadOnly bool `json:"readOnly,omitempty"`
}

// RBDVolumeSource represents a Rados Block Device Mount that lasts the lifetime of a pod
type RBDVolumeSource struct {
	// Required: CephMonitors is a collection of Ceph monitors
	CephMonitors []string `json:"monitors"`
	// Required: RBDImage is the rados image name
	RBDImage string `json:"image"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType,omitempty"`
	// Optional: RadosPool is the rados pool name,default is rbd
	RBDPool string `json:"pool"`
	// Optional: RBDUser is the rados user name, default is admin
	RadosUser string `json:"user"`
	// Optional: Keyring is the path to key ring for RBDUser, default is /etc/ceph/keyring
	Keyring string `json:"keyring"`
	// Optional: SecretRef is name of the authentication secret for RBDUser, default is empty.
	SecretRef *LocalObjectReference `json:"secretRef"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty"`
}

// ContainerPort represents a network port in a single container
type ContainerPort struct {
	// Optional: If specified, this must be an IANA_SVC_NAME  Each named port
	// in a pod must have a unique name.
	Name string `json:"name,omitempty"`
	// Optional: If specified, this must be a valid port number, 0 < x < 65536.
	// If HostNetwork is specified, this must match ContainerPort.
	HostPort int `json:"hostPort,omitempty"`
	// Required: This must be a valid port number, 0 < x < 65536.
	ContainerPort int `json:"containerPort"`
	// Required: Supports "TCP" and "UDP".
	Protocol Protocol `json:"protocol,omitempty"`
	// Optional: What host IP to bind the external port to.
	HostIP string `json:"hostIP,omitempty"`
}

// VolumeMount describes a mounting of a Volume within a container.
type VolumeMount struct {
	// Required: This must match the Name of a Volume [above].
	Name string `json:"name"`
	// Optional: Defaults to false (read-write).
	ReadOnly bool `json:"readOnly,omitempty"`
	// Required.
	MountPath string `json:"mountPath"`
}

// EnvVar represents an environment variable present in a Container.
type EnvVar struct {
	// Required: This must be a C_IDENTIFIER.
	Name string `json:"name"`
	// Optional: no more than one of the following may be specified.
	// Optional: Defaults to ""; variable references $(VAR_NAME) are expanded
	// using the previous defined environment variables in the container and
	// any service environment variables.  If a variable cannot be resolved,
	// the reference in the input string will be unchanged.  The $(VAR_NAME)
	// syntax can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped
	// references will never be expanded, regardless of whether the variable
	// exists or not.
	Value string `json:"value,omitempty"`
	// Optional: Specifies a source the value of this var should come from.
	ValueFrom *EnvVarSource `json:"valueFrom,omitempty"`
}

// EnvVarSource represents a source for the value of an EnvVar.
type EnvVarSource struct {
	// Required: Selects a field of the pod; only name and namespace are supported.
	FieldRef *ObjectFieldSelector `json:"fieldRef"`
}

// ObjectFieldSelector selects an APIVersioned field of an object.
type ObjectFieldSelector struct {
	// Required: Version of the schema the FieldPath is written in terms of.
	// If no value is specified, it will be defaulted to the APIVersion of the
	// enclosing object.
	APIVersion string `json:"apiVersion"`
	// Required: Path of the field to select in the specified API version
	FieldPath string `json:"fieldPath"`
}

// HTTPGetAction describes an action based on HTTP Get requests.
type HTTPGetAction struct {
	// Optional: Path to access on the HTTP server.
	Path string `json:"path,omitempty"`
	// Required: Name or number of the port to access on the container.
	Port util.IntOrString `json:"port,omitempty"`
	// Optional: Host name to connect to, defaults to the pod IP.
	Host string `json:"host,omitempty"`
	// Optional: Scheme to use for connecting to the host, defaults to HTTP.
	Scheme URIScheme `json:"scheme,omitempty"`
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
	Port util.IntOrString `json:"port,omitempty"`
}

// ExecAction describes a "run in container" action.
type ExecAction struct {
	// Command is the command line to execute inside the container, the working directory for the
	// command  is root ('/') in the container's filesystem.  The command is simply exec'd, it is
	// not run inside a shell, so traditional shell instructions ('|', etc) won't work.  To use
	// a shell, you need to explicitly call out to that shell.
	Command []string `json:"command,omitempty"`
}

// Probe describes a liveness probe to be examined to the container.
type Probe struct {
	// The action taken to determine the health of a container
	Handler `json:",inline"`
	// Length of time before health checking is activated.  In seconds.
	InitialDelaySeconds int64 `json:"initialDelaySeconds,omitempty"`
	// Length of time before health checking times out.  In seconds.
	TimeoutSeconds int64 `json:"timeoutSeconds,omitempty"`
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

// Capability represent POSIX capabilities type
type Capability string

// Capabilities represent POSIX capabilities that can be added or removed to a running container.
type Capabilities struct {
	// Added capabilities
	Add []Capability `json:"add,omitempty"`
	// Removed capabilities
	Drop []Capability `json:"drop,omitempty"`
}

// ResourceRequirements describes the compute resource requirements.
type ResourceRequirements struct {
	// Limits describes the maximum amount of compute resources allowed.
	Limits ResourceList `json:"limits,omitempty"`
	// Requests describes the minimum amount of compute resources required.
	// If Request is omitted for a container, it defaults to Limits if that is explicitly specified,
	// otherwise to an implementation-defined value
	Requests ResourceList `json:"requests,omitempty"`
}

// Container represents a single container that is expected to be run on the host.
type Container struct {
	// Required: This must be a DNS_LABEL.  Each container in a pod must
	// have a unique name.
	Name string `json:"name"`
	// Required.
	Image string `json:"image"`
	// Optional: The docker image's entrypoint is used if this is not provided; cannot be updated.
	// Variable references $(VAR_NAME) are expanded using the container's environment.  If a variable
	// cannot be resolved, the reference in the input string will be unchanged.  The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	Command []string `json:"command,omitempty"`
	// Optional: The docker image's cmd is used if this is not provided; cannot be updated.
	// Variable references $(VAR_NAME) are expanded using the container's environment.  If a variable
	// cannot be resolved, the reference in the input string will be unchanged.  The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	Args []string `json:"args,omitempty"`
	// Optional: Defaults to Docker's default.
	WorkingDir string          `json:"workingDir,omitempty"`
	Ports      []ContainerPort `json:"ports,omitempty"`
	Env        []EnvVar        `json:"env,omitempty"`
	// Compute resource requirements.
	Resources      ResourceRequirements `json:"resources,omitempty"`
	VolumeMounts   []VolumeMount        `json:"volumeMounts,omitempty"`
	LivenessProbe  *Probe               `json:"livenessProbe,omitempty"`
	ReadinessProbe *Probe               `json:"readinessProbe,omitempty"`
	Lifecycle      *Lifecycle           `json:"lifecycle,omitempty"`
	// Required.
	TerminationMessagePath string `json:"terminationMessagePath,omitempty"`
	// Required: Policy for pulling images for this container
	ImagePullPolicy PullPolicy `json:"imagePullPolicy"`
	// Optional: SecurityContext defines the security options the pod should be run with
	SecurityContext *SecurityContext `json:"securityContext,omitempty"`

	// Variables for interactive containers, these have very specialized use-cases (e.g. debugging)
	// and shouldn't be used for general purpose containers.
	Stdin bool `json:"stdin,omitempty" description:"Whether this container should allocate a buffer for stdin in the container runtime; default is false"`
	TTY   bool `json:"tty,omitempty" description:"Whether this container should allocate a TTY for itself, also requires 'stdin' to be true; default is false"`
}

// Handler defines a specific action that should be taken
// TODO: pass structured data to these actions, and document that data here.
type Handler struct {
	// One and only one of the following should be specified.
	// Exec specifies the action to take.
	Exec *ExecAction `json:"exec,omitempty"`
	// HTTPGet specifies the http request to perform.
	HTTPGet *HTTPGetAction `json:"httpGet,omitempty"`
	// TCPSocket specifies an action involving a TCP port.
	// TODO: implement a realistic TCP lifecycle hook
	TCPSocket *TCPSocketAction `json:"tcpSocket,omitempty"`
}

// Lifecycle describes actions that the management system should take in response to container lifecycle
// events.  For the PostStart and PreStop lifecycle handlers, management of the container blocks
// until the action is complete, unless the container process fails, in which case the handler is aborted.
type Lifecycle struct {
	// PostStart is called immediately after a container is created.  If the handler fails, the container
	// is terminated and restarted.
	PostStart *Handler `json:"postStart,omitempty"`
	// PreStop is called immediately before a container is terminated.  The reason for termination is
	// passed to the handler.  Regardless of the outcome of the handler, the container is eventually terminated.
	PreStop *Handler `json:"preStop,omitempty"`
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
	// Reason could be pulling image,
	Reason string `json:"reason,omitempty"`
}

type ContainerStateRunning struct {
	StartedAt util.Time `json:"startedAt,omitempty"`
}

type ContainerStateTerminated struct {
	ExitCode    int       `json:"exitCode"`
	Signal      int       `json:"signal,omitempty"`
	Reason      string    `json:"reason,omitempty"`
	Message     string    `json:"message,omitempty"`
	StartedAt   util.Time `json:"startedAt,omitempty"`
	FinishedAt  util.Time `json:"finishedAt,omitempty"`
	ContainerID string    `json:"containerID,omitempty"`
}

// ContainerState holds a possible state of container.
// Only one of its members may be specified.
// If none of them is specified, the default one is ContainerStateWaiting.
type ContainerState struct {
	Waiting    *ContainerStateWaiting    `json:"waiting,omitempty"`
	Running    *ContainerStateRunning    `json:"running,omitempty"`
	Terminated *ContainerStateTerminated `json:"terminated,omitempty"`
}

type ContainerStatus struct {
	// Each container in a pod must have a unique name.
	Name string `json:"name"`
	// TODO(dchen1107): Should we rename PodStatus to a more generic name or have a separate states
	// defined for container?
	State                ContainerState `json:"state,omitempty"`
	LastTerminationState ContainerState `json:"lastState,omitempty"`
	// Ready specifies whether the conatiner has passed its readiness check.
	Ready bool `json:"ready"`
	// Note that this is calculated from dead containers.  But those containers are subject to
	// garbage collection.  This value will get capped at 5 by GC.
	RestartCount int    `json:"restartCount"`
	Image        string `json:"image"`
	ImageID      string `json:"imageID"`
	ContainerID  string `json:"containerID,omitempty"`
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
	// PodReady means the pod is able to service requests and should be added to the
	// load balancing pools of all matching services.
	PodReady PodConditionType = "Ready"
)

// TODO: add LastTransitionTime, Reason, Message to match NodeCondition api.
type PodCondition struct {
	Type   PodConditionType `json:"type"`
	Status ConditionStatus  `json:"status"`
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

// PodList is a list of Pods.
type PodList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Pod `json:"items"`
}

// DNSPolicy defines how a pod's DNS will be configured.
type DNSPolicy string

const (
	// DNSClusterFirst indicates that the pod should use cluster DNS
	// first, if it is available, then fall back on the default (as
	// determined by kubelet) DNS settings.
	DNSClusterFirst DNSPolicy = "ClusterFirst"

	// DNSDefault indicates that the pod should use the default (as
	// determined by kubelet) DNS settings.
	DNSDefault DNSPolicy = "Default"
)

// PodSpec is a description of a pod
type PodSpec struct {
	Volumes []Volume `json:"volumes"`
	// Required: there must be at least one container in a pod.
	Containers    []Container   `json:"containers"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty"`
	// Optional duration in seconds the pod needs to terminate gracefully. May be decreased in delete request.
	// Value must be non-negative integer. The value zero indicates delete immediately.
	// If this value is nil, the default grace period will be used instead.
	// The grace period is the duration in seconds after the processes running in the pod are sent
	// a termination signal and the time when the processes are forcibly halted with a kill signal.
	// Set this value longer than the expected cleanup time for your process.
	TerminationGracePeriodSeconds *int64 `json:"terminationGracePeriodSeconds,omitempty"`
	// Optional duration in seconds relative to the StartTime that the pod may be active on a node
	// before the system actively tries to terminate the pod; value must be positive integer
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty"`
	// Required: Set DNS policy.
	DNSPolicy DNSPolicy `json:"dnsPolicy,omitempty"`
	// NodeSelector is a selector which must be true for the pod to fit on a node
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// ServiceAccountName is the name of the ServiceAccount to use to run this pod
	// The pod will be allowed to use secrets referenced by the ServiceAccount
	ServiceAccountName string `json:"serviceAccountName"`

	// NodeName is a request to schedule this pod onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this pod onto that node, assuming that it fits resource
	// requirements.
	NodeName string `json:"nodeName,omitempty"`
	// Uses the host's network namespace. If this option is set, the ports that will be
	// used must be specified.
	// Optional: Default to false.
	HostNetwork bool `json:"hostNetwork,omitempty"`
	// ImagePullSecrets is an optional list of references to secrets in the same namespace to use for pulling any of the images used by this PodSpec.
	// If specified, these secrets will be passed to individual puller implementations for them to use.  For example,
	// in the case of docker, only DockerConfig type secrets are honored.
	ImagePullSecrets []LocalObjectReference `json:"imagePullSecrets,omitempty"`
}

// PodStatus represents information about the status of a pod. Status may trail the actual
// state of a system.
type PodStatus struct {
	Phase      PodPhase       `json:"phase,omitempty"`
	Conditions []PodCondition `json:"conditions,omitempty"`
	// A human readable message indicating details about why the pod is in this state.
	Message string `json:"message,omitempty"`
	// A brief CamelCase message indicating details about why the pod is in this state. e.g. 'OutOfDisk'
	Reason string `json:"reason,omitempty"`

	HostIP string `json:"hostIP,omitempty"`
	PodIP  string `json:"podIP,omitempty"`

	// Date and time at which the object was acknowledged by the Kubelet.
	// This is before the Kubelet pulled the container image(s) for the pod.
	StartTime *util.Time `json:"startTime,omitempty"`

	// The list has one entry per container in the manifest. Each entry is
	// currently the output of `docker inspect`. This output format is *not*
	// final and should not be relied upon.
	// TODO: Make real decisions about what our info should look like. Re-enable fuzz test
	// when we have done this.
	ContainerStatuses []ContainerStatus `json:"containerStatuses,omitempty"`
}

// PodStatusResult is a wrapper for PodStatus returned by kubelet that can be encode/decoded
type PodStatusResult struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`
	// Status represents the current information about a pod. This data may not be up
	// to date.
	Status PodStatus `json:"status,omitempty"`
}

// Pod is a collection of containers, used as either input (create, update) or as output (list, get).
type Pod struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a pod.
	Spec PodSpec `json:"spec,omitempty"`

	// Status represents the current information about a pod. This data may not be up
	// to date.
	Status PodStatus `json:"status,omitempty"`
}

// PodTemplateSpec describes the data a pod should have when created from a template
type PodTemplateSpec struct {
	// Metadata of the pods created from this template.
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a pod.
	Spec PodSpec `json:"spec,omitempty"`
}

// PodTemplate describes a template for creating copies of a predefined pod.
type PodTemplate struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Template defines the pods that will be created from this pod template
	Template PodTemplateSpec `json:"template,omitempty"`
}

// PodTemplateList is a list of PodTemplates.
type PodTemplateList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []PodTemplate `json:"items"`
}

// ReplicationControllerSpec is the specification of a replication controller.
// As the internal representation of a replication controller, it may have either
// a TemplateRef or a Template set.
type ReplicationControllerSpec struct {
	// Replicas is the number of desired replicas.
	Replicas int `json:"replicas"`

	// Selector is a label query over pods that should match the Replicas count.
	Selector map[string]string `json:"selector"`

	// TemplateRef is a reference to an object that describes the pod that will be created if
	// insufficient replicas are detected. This reference is ignored if a Template is set.
	// Must be set before converting to a versioned API object
	//TemplateRef *ObjectReference `json:"templateRef,omitempty"`

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected. Internally, this takes precedence over a
	// TemplateRef.
	// Must be set before converting to a v1beta1 or v1beta2 API object.
	Template *PodTemplateSpec `json:"template,omitempty"`
}

// ReplicationControllerStatus represents the current status of a replication
// controller.
type ReplicationControllerStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int `json:"replicas"`

	// ObservedGeneration is the most recent generation observed by the controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// ReplicationController represents the configuration of a replication controller.
type ReplicationController struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired behavior of this replication controller.
	Spec ReplicationControllerSpec `json:"spec,omitempty"`

	// Status is the current status of this replication controller. This data may be
	// out of date by some window of time.
	Status ReplicationControllerStatus `json:"status,omitempty"`
}

// ReplicationControllerList is a collection of replication controllers.
type ReplicationControllerList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []ReplicationController `json:"items"`
}

// DaemonSpec is the specification of a daemon.
type DaemonSpec struct {
	// Selector is a label query over pods that are managed by the daemon.
	Selector map[string]string `json:"selector"`

	// Template is the object that describes the pod that will be created.
	// The Daemon will create exactly one copy of this pod on every node
	// that matches the template's node selector (or on every node if no node
	// selector is specified).
	Template *PodTemplateSpec `json:"template,omitempty"`
}

// DaemonStatus represents the current status of a daemon.
type DaemonStatus struct {
	// CurrentNumberScheduled is the number of nodes that are running exactly 1 copy of the
	// daemon and are supposed to run the daemon.
	CurrentNumberScheduled int `json:"currentNumberScheduled"`

	// NumberMisscheduled is the number of nodes that are running the daemon, but are
	// not supposed to run the daemon.
	NumberMisscheduled int `json:"numberMisscheduled"`

	// DesiredNumberScheduled is the total number of nodes that should be running the daemon
	// (including nodes correctly running the daemon).
	DesiredNumberScheduled int `json:"desiredNumberScheduled"`
}

// Daemon represents the configuration of a daemon.
type Daemon struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired behavior of this daemon.
	Spec DaemonSpec `json:"spec,omitempty"`

	// Status is the current status of this daemon. This data may be
	// out of date by some window of time.
	Status DaemonStatus `json:"status,omitempty"`
}

// DaemonList is a collection of daemon.
type DaemonList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Daemon `json:"items"`
}

const (
	// ClusterIPNone - do not assign a cluster IP
	// no proxying required and no environment variables should be created for pods
	ClusterIPNone = "None"
)

// ServiceList holds a list of services.
type ServiceList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Service `json:"items"`
}

// Session Affinity Type string
type ServiceAffinity string

const (
	// ServiceAffinityClientIP is the Client IP based.
	ServiceAffinityClientIP ServiceAffinity = "ClientIP"

	// ServiceAffinityNone - no session affinity.
	ServiceAffinityNone ServiceAffinity = "None"
)

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
)

// ServiceStatus represents the current status of a service
type ServiceStatus struct {
	// LoadBalancer contains the current status of the load-balancer,
	// if one is present.
	LoadBalancer LoadBalancerStatus `json:"loadBalancer,omitempty"`
}

// LoadBalancerStatus represents the status of a load-balancer
type LoadBalancerStatus struct {
	// Ingress is a list containing ingress points for the load-balancer;
	// traffic intended for the service should be sent to these ingress points.
	Ingress []LoadBalancerIngress `json:"ingress,omitempty"`
}

// LoadBalancerIngress represents the status of a load-balancer ingress point:
// traffic intended for the service should be sent to an ingress point.
type LoadBalancerIngress struct {
	// IP is set for load-balancer ingress points that are IP based
	// (typically GCE or OpenStack load-balancers)
	IP string `json:"ip,omitempty"`

	// Hostname is set for load-balancer ingress points that are DNS based
	// (typically AWS load-balancers)
	Hostname string `json:"hostname,omitempty"`
}

// ServiceSpec describes the attributes that a user creates on a service
type ServiceSpec struct {
	// Required: The list of ports that are exposed by this service.
	Ports []ServicePort `json:"ports"`

	// This service will route traffic to pods having labels matching this selector. If empty or not present,
	// the service is assumed to have endpoints set by an external process and Kubernetes will not modify
	// those endpoints.
	Selector map[string]string `json:"selector"`

	// ClusterIP is usually assigned by the master.  If specified by the user
	// we will try to respect it or else fail the request.  This field can
	// not be changed by updates.
	// Valid values are None, empty string (""), or a valid IP address
	// None can be specified for headless services when proxying is not required
	ClusterIP string `json:"clusterIP,omitempty"`

	// Type determines how the service will be exposed.  Valid options: ClusterIP, NodePort, LoadBalancer
	Type ServiceType `json:"type,omitempty"`

	// DeprecatedPublicIPs are deprecated and silently ignored.
	// Old behaviour: PublicIPs are used by external load balancers, or can be set by
	// users to handle external traffic that arrives at a node.
	DeprecatedPublicIPs []string `json:"deprecatedPublicIPs,omitempty"`

	// Required: Supports "ClientIP" and "None".  Used to maintain session affinity.
	SessionAffinity ServiceAffinity `json:"sessionAffinity,omitempty"`
}

type ServicePort struct {
	// Optional if only one ServicePort is defined on this service: The
	// name of this port within the service.  This must be a DNS_LABEL.
	// All ports within a ServiceSpec must have unique names.  This maps to
	// the 'Name' field in EndpointPort objects.
	Name string `json:"name"`

	// The IP protocol for this port.  Supports "TCP" and "UDP".
	Protocol Protocol `json:"protocol"`

	// The port that will be exposed on the service.
	Port int `json:"port"`

	// Optional: The target port on pods selected by this service.  If this
	// is a string, it will be looked up as a named port in the target
	// Pod's container ports.  If this is not specified, the default value
	// is the sames as the Port field (an identity map).
	TargetPort util.IntOrString `json:"targetPort"`

	// The port on each node on which this service is exposed.
	// Default is to auto-allocate a port if the ServiceType of this Service requires one.
	NodePort int `json:"nodePort"`
}

// Service is a named abstraction of software service (for example, mysql) consisting of local port
// (for example 3306) that the proxy listens on, and the selector that determines which pods
// will answer requests sent through the proxy.
type Service struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a service.
	Spec ServiceSpec `json:"spec,omitempty"`

	// Status represents the current status of a service.
	Status ServiceStatus `json:"status,omitempty"`
}

// ServiceAccount binds together:
// * a name, understood by users, and perhaps by peripheral systems, for an identity
// * a principal that can be authenticated and authorized
// * a set of secrets
type ServiceAccount struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Secrets is the list of secrets allowed to be used by pods running using this ServiceAccount
	Secrets []ObjectReference `json:"secrets"`

	// ImagePullSecrets is a list of references to secrets in the same namespace to use for pulling any images
	// in pods that reference this ServiceAccount.  ImagePullSecrets are distinct from Secrets because Secrets
	// can be mounted in the pod, but ImagePullSecrets are only accessed by the kubelet.
	ImagePullSecrets []LocalObjectReference `json:"imagePullSecrets,omitempty"`
}

// ServiceAccountList is a list of ServiceAccount objects
type ServiceAccountList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []ServiceAccount `json:"items"`
}

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
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

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
	Addresses []EndpointAddress
	Ports     []EndpointPort
}

// EndpointAddress is a tuple that describes single IP address.
type EndpointAddress struct {
	// The IP of this endpoint.
	// TODO: This should allow hostname or IP, see #4447.
	IP string

	// Optional: The kubernetes object related to the entry point.
	TargetRef *ObjectReference
}

// EndpointPort is a tuple that describes a single port.
type EndpointPort struct {
	// The name of this port (corresponds to ServicePort.Name).  Optional
	// if only one port is defined.  Must be a DNS_LABEL.
	Name string

	// The port number.
	Port int

	// The IP protocol for this port.
	Protocol Protocol
}

// EndpointsList is a list of endpoints.
type EndpointsList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Endpoints `json:"items"`
}

// NodeSpec describes the attributes that a node is created with.
type NodeSpec struct {
	// PodCIDR represents the pod IP range assigned to the node
	// Note: assigning IP ranges to nodes might need to be revisited when we support migratable IPs.
	PodCIDR string `json:"podCIDR,omitempty"`

	// External ID of the node assigned by some machine database (e.g. a cloud provider)
	ExternalID string `json:"externalID,omitempty"`

	// ID of the node assigned by the cloud provider
	// Note: format is "<ProviderName>://<ProviderSpecificNodeID>"
	ProviderID string `json:"providerID,omitempty"`

	// Unschedulable controls node schedulability of new pods. By default node is schedulable.
	Unschedulable bool `json:"unschedulable,omitempty"`
}

// NodeSystemInfo is a set of ids/uuids to uniquely identify the node.
type NodeSystemInfo struct {
	// MachineID is the machine-id reported by the node
	MachineID string `json:"machineID"`
	// SystemUUID is the system-uuid reported by the node
	SystemUUID string `json:"systemUUID"`
	// BootID is the boot-id reported by the node
	BootID string `json:"bootID"`
	// Kernel version reported by the node
	KernelVersion string `json:"kernelVersion"`
	// OS image used reported by the node
	OsImage string `json:"osImage"`
	// Container runtime version reported by the node
	ContainerRuntimeVersion string `json:"containerRuntimeVersion"`
	// Kubelet version reported by the node
	KubeletVersion string `json:"kubeletVersion"`
	// Kube-proxy version reported by the node
	KubeProxyVersion string `json:"kubeProxyVersion"`
}

// NodeStatus is information about the current status of a node.
type NodeStatus struct {
	// Capacity represents the available resources of a node.
	Capacity ResourceList `json:"capacity,omitempty"`
	// NodePhase is the current lifecycle phase of the node.
	Phase NodePhase `json:"phase,omitempty"`
	// Conditions is an array of current node conditions.
	Conditions []NodeCondition `json:"conditions,omitempty"`
	// Queried from cloud provider, if available.
	Addresses []NodeAddress `json:"addresses,omitempty"`
	// NodeSystemInfo is a set of ids/uuids to uniquely identify the node
	NodeInfo NodeSystemInfo `json:"nodeInfo,omitempty"`
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
)

type NodeCondition struct {
	Type               NodeConditionType `json:"type"`
	Status             ConditionStatus   `json:"status"`
	LastHeartbeatTime  util.Time         `json:"lastHeartbeatTime,omitempty"`
	LastTransitionTime util.Time         `json:"lastTransitionTime,omitempty"`
	Reason             string            `json:"reason,omitempty"`
	Message            string            `json:"message,omitempty"`
}

type NodeAddressType string

// These are valid address types of node. NodeLegacyHostIP is used to transit
// from out-dated HostIP field to NodeAddress.
const (
	NodeLegacyHostIP NodeAddressType = "LegacyHostIP"
	NodeHostName     NodeAddressType = "Hostname"
	NodeExternalIP   NodeAddressType = "ExternalIP"
	NodeInternalIP   NodeAddressType = "InternalIP"
)

type NodeAddress struct {
	Type    NodeAddressType `json:"type"`
	Address string          `json:"address"`
}

// NodeResources is an object for conveying resource information about a node.
// see http://docs.k8s.io/design/resources.md for more details.
type NodeResources struct {
	// Capacity represents the available resources of a node
	Capacity ResourceList `json:"capacity,omitempty"`
}

// ResourceName is the name identifying various resources in a ResourceList.
type ResourceName string

const (
	// CPU, in cores. (500m = .5 cores)
	ResourceCPU ResourceName = "cpu"
	// Memory, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceMemory ResourceName = "memory"
	// Volume size, in bytes (e,g. 5Gi = 5GiB = 5 * 1024 * 1024 * 1024)
	ResourceStorage ResourceName = "storage"
	// Number of Pods that may be running on this Node: see ResourcePods
)

// ResourceList is a set of (resource name, quantity) pairs.
type ResourceList map[ResourceName]resource.Quantity

// Node is a worker node in Kubernetes
// The name of the node according to etcd is in ObjectMeta.Name.
type Node struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a node.
	Spec NodeSpec `json:"spec,omitempty"`

	// Status describes the current status of a Node
	Status NodeStatus `json:"status,omitempty"`
}

// NodeList is a list of nodes.
type NodeList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Node `json:"items"`
}

// NamespaceSpec describes the attributes on a Namespace
type NamespaceSpec struct {
	// Finalizers is an opaque list of values that must be empty to permanently remove object from storage
	Finalizers []FinalizerName
}

type FinalizerName string

// These are internal finalizer values to Kubernetes, must be qualified name unless defined here
const (
	FinalizerKubernetes FinalizerName = "kubernetes"
)

// NamespaceStatus is information about the current status of a Namespace.
type NamespaceStatus struct {
	// Phase is the current lifecycle phase of the namespace.
	Phase NamespacePhase `json:"phase,omitempty"`
}

type NamespacePhase string

// These are the valid phases of a namespace.
const (
	// NamespaceActive means the namespace is available for use in the system
	NamespaceActive NamespacePhase = "Active"
	// NamespaceTerminating means the namespace is undergoing graceful termination
	NamespaceTerminating NamespacePhase = "Terminating"
)

// A namespace provides a scope for Names.
// Use of multiple namespaces is optional
type Namespace struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of the Namespace.
	Spec NamespaceSpec `json:"spec,omitempty"`

	// Status describes the current status of a Namespace
	Status NamespaceStatus `json:"status,omitempty"`
}

// NamespaceList is a list of Namespaces.
type NamespaceList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Namespace `json:"items"`
}

// Binding ties one object to another - for example, a pod is bound to a node by a scheduler.
type Binding struct {
	TypeMeta `json:",inline"`
	// ObjectMeta describes the object that is being bound.
	ObjectMeta `json:"metadata,omitempty"`

	// Target is the object to bind to.
	Target ObjectReference `json:"target"`
}

// DeleteOptions may be provided when deleting an API object
type DeleteOptions struct {
	TypeMeta `json:",inline"`

	// Optional duration in seconds before the object should be deleted. Value must be non-negative integer.
	// The value zero indicates delete immediately. If this value is nil, the default grace period for the
	// specified type will be used.
	GracePeriodSeconds *int64 `json:"gracePeriodSeconds"`
}

// ListOptions is the query options to a standard REST list call, and has future support for
// watch calls.
type ListOptions struct {
	TypeMeta `json:",inline"`

	// A selector based on labels
	LabelSelector labels.Selector
	// A selector based on fields
	FieldSelector fields.Selector
	// If true, watch for changes to this list
	Watch bool
	// The resource version to watch (no effect on list yet)
	ResourceVersion string
}

// PodLogOptions is the query options for a Pod's logs REST call
type PodLogOptions struct {
	TypeMeta

	// Container for which to return logs
	Container string

	// If true, follow the logs for the pod
	Follow bool

	// If true, return previous terminated container logs
	Previous bool
}

// PodAttachOptions is the query options to a Pod's remote attach call
// TODO: merge w/ PodExecOptions below for stdin, stdout, etc
type PodAttachOptions struct {
	TypeMeta `json:",inline"`

	// Stdin if true indicates that stdin is to be redirected for the attach call
	Stdin bool `json:"stdin,omitempty"`

	// Stdout if true indicates that stdout is to be redirected for the attach call
	Stdout bool `json:"stdout,omitempty"`

	// Stderr if true indicates that stderr is to be redirected for the attach call
	Stderr bool `json:"stderr,omitempty"`

	// TTY if true indicates that a tty will be allocated for the attach call
	TTY bool `json:"tty,omitempty"`

	// Container to attach to.
	Container string `json:"container,omitempty"`
}

// PodExecOptions is the query options to a Pod's remote exec call
type PodExecOptions struct {
	TypeMeta

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

// PodProxyOptions is the query options to a Pod's proxy call
type PodProxyOptions struct {
	TypeMeta

	// Path is the URL path to use for the current proxy request
	Path string
}

// Status is a return value for calls that don't return other objects.
// TODO: this could go in apiserver, but I'm including it here so clients needn't
// import both.
type Status struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	// One of: "Success" or "Failure"
	Status string `json:"status,omitempty"`
	// A human-readable description of the status of this operation.
	Message string `json:"message,omitempty"`
	// A machine-readable description of why this operation is in the
	// "Failure" status. If this value is empty there
	// is no information available. A Reason clarifies an HTTP status
	// code but does not override it.
	Reason StatusReason `json:"reason,omitempty"`
	// Extended data associated with the reason.  Each reason may define its
	// own extended details. This field is optional and the data returned
	// is not guaranteed to conform to any schema except that defined by
	// the reason type.
	Details *StatusDetails `json:"details,omitempty"`
	// Suggested HTTP return code for this status, 0 if not set.
	Code int `json:"code,omitempty"`
}

// StatusDetails is a set of additional properties that MAY be set by the
// server to provide additional information about a response. The Reason
// field of a Status object defines what attributes will be set. Clients
// must ignore fields that do not match the defined type of each attribute,
// and should assume that any attribute may be empty, invalid, or under
// defined.
type StatusDetails struct {
	// The name attribute of the resource associated with the status StatusReason
	// (when there is a single name which can be described).
	Name string `json:"name,omitempty"`
	// The kind attribute of the resource associated with the status StatusReason.
	// On some operations may differ from the requested resource Kind.
	Kind string `json:"kind,omitempty"`
	// The Causes array includes more details associated with the StatusReason
	// failure. Not all StatusReasons may provide detailed causes.
	Causes []StatusCause `json:"causes,omitempty"`
	// If specified, the time in seconds before the operation should be retried.
	RetryAfterSeconds int `json:"retryAfterSeconds,omitempty"`
}

// Values of Status.Status
const (
	StatusSuccess = "Success"
	StatusFailure = "Failure"
)

// StatusReason is an enumeration of possible failure causes.  Each StatusReason
// must map to a single HTTP status code, but multiple reasons may map
// to the same HTTP status code.
// TODO: move to apiserver
type StatusReason string

const (
	// StatusReasonUnknown means the server has declined to indicate a specific reason.
	// The details field may contain other information about this error.
	// Status code 500.
	StatusReasonUnknown StatusReason = ""

	// StatusReasonUnauthorized means the server can be reached and understood the request, but requires
	// the user to present appropriate authorization credentials (identified by the WWW-Authenticate header)
	// in order for the action to be completed. If the user has specified credentials on the request, the
	// server considers them insufficient.
	// Status code 401
	StatusReasonUnauthorized StatusReason = "Unauthorized"

	// StatusReasonForbidden means the server can be reached and understood the request, but refuses
	// to take any further action.  It is the result of the server being configured to deny access for some reason
	// to the requested resource by the client.
	// Details (optional):
	//   "kind" string - the kind attribute of the forbidden resource
	//                   on some operations may differ from the requested
	//                   resource.
	//   "id"   string - the identifier of the forbidden resource
	// Status code 403
	StatusReasonForbidden StatusReason = "Forbidden"

	// StatusReasonNotFound means one or more resources required for this operation
	// could not be found.
	// Details (optional):
	//   "kind" string - the kind attribute of the missing resource
	//                   on some operations may differ from the requested
	//                   resource.
	//   "id"   string - the identifier of the missing resource
	// Status code 404
	StatusReasonNotFound StatusReason = "NotFound"

	// StatusReasonAlreadyExists means the resource you are creating already exists.
	// Details (optional):
	//   "kind" string - the kind attribute of the conflicting resource
	//   "id"   string - the identifier of the conflicting resource
	// Status code 409
	StatusReasonAlreadyExists StatusReason = "AlreadyExists"

	// StatusReasonConflict means the requested update operation cannot be completed
	// due to a conflict in the operation. The client may need to alter the request.
	// Each resource may define custom details that indicate the nature of the
	// conflict.
	// Status code 409
	StatusReasonConflict StatusReason = "Conflict"

	// StatusReasonInvalid means the requested create or update operation cannot be
	// completed due to invalid data provided as part of the request. The client may
	// need to alter the request. When set, the client may use the StatusDetails
	// message field as a summary of the issues encountered.
	// Details (optional):
	//   "kind" string - the kind attribute of the invalid resource
	//   "id"   string - the identifier of the invalid resource
	//   "causes"      - one or more StatusCause entries indicating the data in the
	//                   provided resource that was invalid.  The code, message, and
	//                   field attributes will be set.
	// Status code 422
	StatusReasonInvalid StatusReason = "Invalid"

	// StatusReasonServerTimeout means the server can be reached and understood the request,
	// but cannot complete the action in a reasonable time. The client should retry the request.
	// This is may be due to temporary server load or a transient communication issue with
	// another server. Status code 500 is used because the HTTP spec provides no suitable
	// server-requested client retry and the 5xx class represents actionable errors.
	// Details (optional):
	//   "kind" string - the kind attribute of the resource being acted on.
	//   "id"   string - the operation that is being attempted.
	//   "retryAfterSeconds" int - the number of seconds before the operation should be retried
	// Status code 500
	StatusReasonServerTimeout StatusReason = "ServerTimeout"

	// StatusReasonTimeout means that the request could not be completed within the given time.
	// Clients can get this response only when they specified a timeout param in the request,
	// or if the server cannot complete the operation within a reasonable amount of time.
	// The request might succeed with an increased value of timeout param. The client *should*
	// wait at least the number of seconds specified by the retryAfterSeconds field.
	// Details (optional):
	//   "retryAfterSeconds" int - the number of seconds before the operation should be retried
	// Status code 504
	StatusReasonTimeout StatusReason = "Timeout"

	// StatusReasonBadRequest means that the request itself was invalid, because the request
	// doesn't make any sense, for example deleting a read-only object.  This is different than
	// StatusReasonInvalid above which indicates that the API call could possibly succeed, but the
	// data was invalid.  API calls that return BadRequest can never succeed.
	StatusReasonBadRequest StatusReason = "BadRequest"

	// StatusReasonMethodNotAllowed means that the action the client attempted to perform on the
	// resource was not supported by the code - for instance, attempting to delete a resource that
	// can only be created. API calls that return MethodNotAllowed can never succeed.
	StatusReasonMethodNotAllowed StatusReason = "MethodNotAllowed"

	// StatusReasonInternalError indicates that an internal error occurred, it is unexpected
	// and the outcome of the call is unknown.
	// Details (optional):
	//   "causes" - The original error
	// Status code 500
	StatusReasonInternalError = "InternalError"

	// StatusReasonServiceUnavailable means that the request itself was valid,
	// but the requested service is unavailable at this time.
	// Retrying the request after some time might succeed.
	// Status code 503
	StatusReasonServiceUnavailable StatusReason = "ServiceUnavailable"
)

// StatusCause provides more information about an api.Status failure, including
// cases when multiple errors are encountered.
type StatusCause struct {
	// A machine-readable description of the cause of the error. If this value is
	// empty there is no information available.
	Type CauseType `json:"reason,omitempty"`
	// A human-readable description of the cause of the error.  This field may be
	// presented as-is to a reader.
	Message string `json:"message,omitempty"`
	// The field of the resource that has caused this error, as named by its JSON
	// serialization. May include dot and postfix notation for nested attributes.
	// Arrays are zero-indexed.  Fields may appear more than once in an array of
	// causes due to fields having multiple errors.
	// Optional.
	//
	// Examples:
	//   "name" - the field "name" on the current resource
	//   "items[0].name" - the field "name" on the first array entry in "items"
	Field string `json:"field,omitempty"`
}

// CauseType is a machine readable value providing more detail about what
// occurred in a status response. An operation may have multiple causes for a
// status (whether Failure or Success).
type CauseType string

const (
	// CauseTypeFieldValueNotFound is used to report failure to find a requested value
	// (e.g. looking up an ID).
	CauseTypeFieldValueNotFound CauseType = "FieldValueNotFound"
	// CauseTypeFieldValueRequired is used to report required values that are not
	// provided (e.g. empty strings, null values, or empty arrays).
	CauseTypeFieldValueRequired CauseType = "FieldValueRequired"
	// CauseTypeFieldValueDuplicate is used to report collisions of values that must be
	// unique (e.g. unique IDs).
	CauseTypeFieldValueDuplicate CauseType = "FieldValueDuplicate"
	// CauseTypeFieldValueInvalid is used to report malformed values (e.g. failed regex
	// match).
	CauseTypeFieldValueInvalid CauseType = "FieldValueInvalid"
	// CauseTypeFieldValueNotSupported is used to report valid (as per formatting rules)
	// values that can not be handled (e.g. an enumerated string).
	CauseTypeFieldValueNotSupported CauseType = "FieldValueNotSupported"
	// CauseTypeUnexpectedServerResponse is used to report when the server responded to the client
	// without the expected return type. The presence of this cause indicates the error may be
	// due to an intervening proxy or the server software malfunctioning.
	CauseTypeUnexpectedServerResponse CauseType = "UnexpectedServerResponse"
)

// ObjectReference contains enough information to let you inspect or modify the referred object.
type ObjectReference struct {
	Kind            string    `json:"kind,omitempty"`
	Namespace       string    `json:"namespace,omitempty"`
	Name            string    `json:"name,omitempty"`
	UID             types.UID `json:"uid,omitempty"`
	APIVersion      string    `json:"apiVersion,omitempty"`
	ResourceVersion string    `json:"resourceVersion,omitempty"`

	// Optional. If referring to a piece of an object instead of an entire object, this string
	// should contain information to identify the sub-object. For example, if the object
	// reference is to a container within a pod, this would take on a value like:
	// "spec.containers{name}" (where "name" refers to the name of the container that triggered
	// the event) or if no container name is specified "spec.containers[2]" (container with
	// index 2 in this pod). This syntax is chosen only to have some well-defined way of
	// referencing a part of an object.
	// TODO: this design is not final and this field is subject to change in the future.
	FieldPath string `json:"fieldPath,omitempty"`
}

// LocalObjectReference contains enough information to let you locate the referenced object inside the same namespace.
type LocalObjectReference struct {
	//TODO: Add other useful fields.  apiVersion, kind, uid?
	Name string
}

type SerializedReference struct {
	TypeMeta  `json:",inline"`
	Reference ObjectReference `json:"reference,omitempty"`
}

type EventSource struct {
	// Component from which the event is generated.
	Component string `json:"component,omitempty"`
	// Host name on which the event is generated.
	Host string `json:"host,omitempty"`
}

// Event is a report of an event somewhere in the cluster.
// TODO: Decide whether to store these separately or with the object they apply to.
type Event struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Required. The object that this event is about.
	InvolvedObject ObjectReference `json:"involvedObject,omitempty"`

	// Optional; this should be a short, machine understandable string that gives the reason
	// for this event being generated. For example, if the event is reporting that a container
	// can't start, the Reason might be "ImageNotFound".
	// TODO: provide exact specification for format.
	Reason string `json:"reason,omitempty"`

	// Optional. A human-readable description of the status of this operation.
	// TODO: decide on maximum length.
	Message string `json:"message,omitempty"`

	// Optional. The component reporting this event. Should be a short machine understandable string.
	Source EventSource `json:"source,omitempty"`

	// The time at which the event was first recorded. (Time of server receipt is in TypeMeta.)
	FirstTimestamp util.Time `json:"firstTimestamp,omitempty"`

	// The time at which the most recent occurrence of this event was recorded.
	LastTimestamp util.Time `json:"lastTimestamp,omitempty"`

	// The number of times this event has occurred.
	Count int `json:"count,omitempty"`
}

// EventList is a list of events.
type EventList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Event `json:"items"`
}

// List holds a list of objects, which may not be known by the server.
type List struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []runtime.Object `json:"items"`
}

// A type of object that is limited
type LimitType string

const (
	// Limit that applies to all pods in a namespace
	LimitTypePod LimitType = "Pod"
	// Limit that applies to all containers in a namespace
	LimitTypeContainer LimitType = "Container"
)

// LimitRangeItem defines a min/max usage limit for any resource that matches on kind
type LimitRangeItem struct {
	// Type of resource that this limit applies to
	Type LimitType `json:"type,omitempty"`
	// Max usage constraints on this kind by resource name
	Max ResourceList `json:"max,omitempty"`
	// Min usage constraints on this kind by resource name
	Min ResourceList `json:"min,omitempty"`
	// Default usage constraints on this kind by resource name
	Default ResourceList `json:"default,omitempty"`
}

// LimitRangeSpec defines a min/max usage limit for resources that match on kind
type LimitRangeSpec struct {
	// Limits is the list of LimitRangeItem objects that are enforced
	Limits []LimitRangeItem `json:"limits"`
}

// LimitRange sets resource usage limits for each kind of resource in a Namespace
type LimitRange struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the limits enforced
	Spec LimitRangeSpec `json:"spec,omitempty"`
}

// LimitRangeList is a list of LimitRange items.
type LimitRangeList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	// Items is a list of LimitRange objects
	Items []LimitRange `json:"items"`
}

// The following identify resource constants for Kubernetes object types
const (
	// Pods, number
	ResourcePods ResourceName = "pods"
	// Services, number
	ResourceServices ResourceName = "services"
	// ReplicationControllers, number
	ResourceReplicationControllers ResourceName = "replicationcontrollers"
	// Daemon, number
	ResourceDaemon ResourceName = "daemon"
	// ResourceQuotas, number
	ResourceQuotas ResourceName = "resourcequotas"
	// ResourceSecrets, number
	ResourceSecrets ResourceName = "secrets"
	// ResourcePersistentVolumeClaims, number
	ResourcePersistentVolumeClaims ResourceName = "persistentvolumeclaims"
)

// ResourceQuotaSpec defines the desired hard limits to enforce for Quota
type ResourceQuotaSpec struct {
	// Hard is the set of desired hard limits for each named resource
	Hard ResourceList `json:"hard,omitempty"`
}

// ResourceQuotaStatus defines the enforced hard limits and observed use
type ResourceQuotaStatus struct {
	// Hard is the set of enforced hard limits for each named resource
	Hard ResourceList `json:"hard,omitempty"`
	// Used is the current observed total usage of the resource in the namespace
	Used ResourceList `json:"used,omitempty"`
}

// ResourceQuota sets aggregate quota restrictions enforced per namespace
type ResourceQuota struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired quota
	Spec ResourceQuotaSpec `json:"spec,omitempty"`

	// Status defines the actual enforced quota and its current usage
	Status ResourceQuotaStatus `json:"status,omitempty"`
}

// ResourceQuotaList is a list of ResourceQuota items
type ResourceQuotaList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	// Items is a list of ResourceQuota objects
	Items []ResourceQuota `json:"items"`
}

// Secret holds secret data of a certain type.  The total bytes of the values in
// the Data field must be less than MaxSecretSize bytes.
type Secret struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Data contains the secret data.  Each key must be a valid DNS_SUBDOMAIN
	// or leading dot followed by valid DNS_SUBDOMAIN.
	// The serialized form of the secret data is a base64 encoded string,
	// representing the arbitrary (possibly non-string) data value here.
	Data map[string][]byte `json:"data,omitempty"`

	// Used to facilitate programmatic handling of secret data.
	Type SecretType `json:"type,omitempty"`
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

	// SecretTypeDockercfg contains a dockercfg file that follows the same format rules as ~/.dockercfg
	//
	// Required fields:
	// - Secret.Data[".dockercfg"] - a serialized ~/.dockercfg file
	SecretTypeDockercfg SecretType = "kubernetes.io/dockercfg"

	// DockerConfigKey is the key of the required data for SecretTypeDockercfg secrets
	DockerConfigKey = ".dockercfg"
)

type SecretList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Secret `json:"items"`
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
	ExecCommandParamm = "command"

	StreamType       = "streamType"
	StreamTypeStdin  = "stdin"
	StreamTypeStdout = "stdout"
	StreamTypeStderr = "stderr"
	StreamTypeData   = "data"
	StreamTypeError  = "error"

	PortHeader = "port"
)

// Similarly to above, these are constants to support HTTP PATCH utilized by
// both the client and server that didn't make sense for a whole package to be
// dedicated to.
type PatchType string

const (
	JSONPatchType           PatchType = "application/json-patch+json"
	MergePatchType          PatchType = "application/merge-patch+json"
	StrategicMergePatchType PatchType = "application/strategic-merge-patch+json"
)

// Type and constants for component health validation.
type ComponentConditionType string

// These are the valid conditions for the component.
const (
	ComponentHealthy ComponentConditionType = "Healthy"
)

type ComponentCondition struct {
	Type    ComponentConditionType `json:"type"`
	Status  ConditionStatus        `json:"status"`
	Message string                 `json:"message,omitempty"`
	Error   string                 `json:"error,omitempty"`
}

// ComponentStatus (and ComponentStatusList) holds the cluster validation info.
type ComponentStatus struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	Conditions []ComponentCondition `json:"conditions,omitempty"`
}

type ComponentStatusList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []ComponentStatus `json:"items"`
}

// SecurityContext holds security configuration that will be applied to a container.  SecurityContext
// contains duplication of some existing fields from the Container resource.  These duplicate fields
// will be populated based on the Container configuration if they are not set.  Defining them on
// both the Container AND the SecurityContext will result in an error.
type SecurityContext struct {
	// Capabilities are the capabilities to add/drop when running the container
	Capabilities *Capabilities `json:"capabilities,omitempty"`

	// Run the container in privileged mode
	Privileged *bool `json:"privileged,omitempty"`

	// SELinuxOptions are the labels to be applied to the container
	// and volumes
	SELinuxOptions *SELinuxOptions `json:"seLinuxOptions,omitempty"`

	// RunAsUser is the UID to run the entrypoint of the container process.
	RunAsUser *int64 `json:"runAsUser,omitempty"`

	// RunAsNonRoot indicates that the container should be run as a non-root user.  If the RunAsUser
	// field is not explicitly set then the kubelet may check the image for a specified user or
	// perform defaulting to specify a user.
	RunAsNonRoot bool
}

// SELinuxOptions are the labels to be applied to the container.
type SELinuxOptions struct {
	// SELinux user label
	User string `json:"user,omitempty"`

	// SELinux role label
	Role string `json:"role,omitempty"`

	// SELinux type label
	Type string `json:"type,omitempty"`

	// SELinux level label.
	Level string `json:"level,omitempty"`
}

// RangeAllocation is an opaque API object (not exposed to end users) that can be persisted to record
// the global allocation state of the cluster. The schema of Range and Data generic, in that Range
// should be a string representation of the inputs to a range (for instance, for IP allocation it
// might be a CIDR) and Data is an opaque blob understood by an allocator which is typically a
// binary range.  Consumers should use annotations to record additional information (schema version,
// data encoding hints). A range allocation should *ALWAYS* be recreatable at any time by observation
// of the cluster, thus the object is less strongly typed than most.
type RangeAllocation struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`
	// A string representing a unique label for a range of resources, such as a CIDR "10.0.0.0/8" or
	// port range "10000-30000". Range is not strongly schema'd here. The Range is expected to define
	// a start and end unless there is an implicit end.
	Range string `json:"range"`
	// A byte array representing the serialized state of a range allocation. Additional clarifiers on
	// the type or format of data should be represented with annotations. For IP allocations, this is
	// represented as a bit array starting at the base IP of the CIDR in Range, with each bit representing
	// a single allocated address (the fifth bit on CIDR 10.0.0.0/8 is 10.0.0.4).
	Data []byte `json:"data"`
}
