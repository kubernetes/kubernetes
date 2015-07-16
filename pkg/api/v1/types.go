/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	Kind string `json:"kind,omitempty" description:"kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#types-kinds"`

	// APIVersion defines the versioned schema of this representation of an object.
	// Servers should convert recognized schemas to the latest internal value, and
	// may reject unrecognized values.
	APIVersion string `json:"apiVersion,omitempty" description:"version of the schema the object should have; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#resources"`
}

// ListMeta describes metadata that synthetic resources must have, including lists and
// various status objects.
type ListMeta struct {
	// SelfLink is a URL representing this object.
	SelfLink string `json:"selfLink,omitempty" description:"URL for the object; populated by the system, read-only"`

	// An opaque value that represents the version of this response for use with optimistic
	// concurrency and change monitoring endpoints.  Clients must treat these values as opaque
	// and values may only be valid for a particular resource or set of resources. Only servers
	// will generate resource versions.
	ResourceVersion string `json:"resourceVersion,omitempty" description:"string that identifies the internal version of this object that can be used by clients to determine when objects have changed; populated by the system, read-only; value must be treated as opaque by clients and passed unmodified back to the server: http://releases.k8s.io/v1.0.0/docs/api-conventions.md#concurrency-control-and-consistency"`
}

// ObjectMeta is metadata that all persisted resources must have, which includes all objects
// users must create.
type ObjectMeta struct {
	// Name is unique within a namespace.  Name is required when creating resources, although
	// some resources may allow a client to request the generation of an appropriate name
	// automatically. Name is primarily intended for creation idempotence and configuration
	// definition.
	Name string `json:"name,omitempty" description:"string that identifies an object. Must be unique within a namespace; cannot be updated; see http://releases.k8s.io/v1.0.0/docs/identifiers.md#names"`

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
	GenerateName string `json:"generateName,omitempty" description:"an optional prefix to use to generate a unique name; has the same validation rules as name; optional, and is applied only name if is not specified; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#idempotency"`

	// Namespace defines the space within which name must be unique. An empty namespace is
	// equivalent to the "default" namespace, but "default" is the canonical representation.
	// Not all objects are required to be scoped to a namespace - the value of this field for
	// those objects will be empty.
	Namespace string `json:"namespace,omitempty" description:"namespace of the object; must be a DNS_LABEL; cannot be updated; see http://releases.k8s.io/v1.0.0/docs/namespaces.md"`

	// SelfLink is a URL representing this object.
	SelfLink string `json:"selfLink,omitempty" description:"URL for the object; populated by the system, read-only"`

	// UID is the unique in time and space value for this object. It is typically generated by
	// the server on successful creation of a resource and is not allowed to change on PUT
	// operations.
	UID types.UID `json:"uid,omitempty" description:"unique UUID across space and time; populated by the system; read-only; see http://releases.k8s.io/v1.0.0/docs/identifiers.md#uids"`

	// An opaque value that represents the version of this resource. May be used for optimistic
	// concurrency, change detection, and the watch operation on a resource or set of resources.
	// Clients must treat these values as opaque and values may only be valid for a particular
	// resource or set of resources. Only servers will generate resource versions.
	ResourceVersion string `json:"resourceVersion,omitempty" description:"string that identifies the internal version of this object that can be used by clients to determine when objects have changed; populated by the system, read-only; value must be treated as opaque by clients and passed unmodified back to the server: http://releases.k8s.io/v1.0.0/docs/api-conventions.md#concurrency-control-and-consistency"`

	// A sequence number representing a specific generation of the desired state.
	// Currently only implemented by replication controllers.
	Generation int64 `json:"generation,omitempty" description:"a sequence number representing a specific generation of the desired state; populated by the system; read-only"`

	// CreationTimestamp is a timestamp representing the server time when this object was
	// created. It is not guaranteed to be set in happens-before order across separate operations.
	// Clients may not set this value. It is represented in RFC3339 form and is in UTC.
	CreationTimestamp util.Time `json:"creationTimestamp,omitempty" description:"RFC 3339 date and time at which the object was created; populated by the system, read-only; null for lists; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// DeletionTimestamp is the time after which this resource will be deleted. This
	// field is set by the server when a graceful deletion is requested by the user, and is not
	// directly settable by a client. The resource will be deleted (no longer visible from
	// resource lists, and not reachable by name) after the time in this field. Once set, this
	// value may not be unset or be set further into the future, although it may be shortened
	// or the resource may be deleted prior to this time. For example, a user may request that
	// a pod is deleted in 30 seconds. The Kubelet will react by sending a graceful termination
	// signal to the containers in the pod. Once the resource is deleted in the API, the Kubelet
	// will send a hard termination signal to the container.
	DeletionTimestamp *util.Time `json:"deletionTimestamp,omitempty" description:"RFC 3339 date and time at which the object will be deleted; populated by the system when a graceful deletion is requested, read-only; if not set, graceful deletion of the object has not been requested; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Labels are key value pairs that may be used to scope and select individual resources.
	// TODO: replace map[string]string with labels.LabelSet type
	Labels map[string]string `json:"labels,omitempty" description:"map of string keys and values that can be used to organize and categorize objects; may match selectors of replication controllers and services; see http://releases.k8s.io/v1.0.0/docs/labels.md"`

	// Annotations are unstructured key value data stored with a resource that may be set by
	// external tooling. They are not queryable and should be preserved when modifying
	// objects.
	Annotations map[string]string `json:"annotations,omitempty" description:"map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about objects; see http://releases.k8s.io/v1.0.0/docs/annotations.md"`
}

const (
	// NamespaceDefault means the object is in the default namespace which is applied when not specified by clients
	NamespaceDefault string = "default"
	// NamespaceAll is the default argument to specify on a context when you want to list or filter resources across all namespaces
	NamespaceAll string = ""
)

// Volume represents a named volume in a pod that may be accessed by any containers in the pod.
type Volume struct {
	// Required: This must be a DNS_LABEL.  Each volume in a pod must have
	// a unique name.
	Name string `json:"name" description:"volume name; must be a DNS_LABEL and unique within the pod; see http://releases.k8s.io/v1.0.0/docs/identifiers.md#names"`
	// Source represents the location and type of a volume to mount.
	// This is optional for now. If not specified, the Volume is implied to be an EmptyDir.
	// This implied behavior is deprecated and will be removed in a future version.
	VolumeSource `json:",inline"`
}

// VolumeSource represents the source location of a volume to mount.
// Only one of its members may be specified.
type VolumeSource struct {
	// HostPath represents a pre-existing file or directory on the host
	// machine that is directly exposed to the container. This is generally
	// used for system agents or other privileged things that are allowed
	// to see the host machine. Most containers will NOT need this.
	// TODO(jonesdl) We need to restrict who can use host directory mounts and who can/can not
	// mount host directories as read/write.
	HostPath *HostPathVolumeSource `json:"hostPath,omitempty" description:"pre-existing host file or directory; generally for privileged system daemons or other agents tied to the host; see http://releases.k8s.io/v1.0.0/docs/volumes.md#hostpath"`
	// EmptyDir represents a temporary directory that shares a pod's lifetime.
	EmptyDir *EmptyDirVolumeSource `json:"emptyDir,omitempty" description:"temporary directory that shares a pod's lifetime; see http://releases.k8s.io/v1.0.0/docs/volumes.md#emptydir"`
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDiskVolumeSource `json:"gcePersistentDisk,omitempty" description:"GCE disk resource attached to the host machine on demand; see http://releases.k8s.io/v1.0.0/docs/volumes.md#gcepersistentdisk"`
	// AWSElasticBlockStore represents an AWS Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	AWSElasticBlockStore *AWSElasticBlockStoreVolumeSource `json:"awsElasticBlockStore,omitempty" description:"AWS disk resource attached to the host machine on demand; see http://releases.k8s.io/v1.0.0/docs/volumes.md#awselasticblockstore"`
	// GitRepo represents a git repository at a particular revision.
	GitRepo *GitRepoVolumeSource `json:"gitRepo,omitempty" description:"git repository at a particular revision"`
	// Secret represents a secret that should populate this volume.
	Secret *SecretVolumeSource `json:"secret,omitempty" description:"secret to populate volume; see http://releases.k8s.io/v1.0.0/docs/volumes.md#secrets"`
	// NFS represents an NFS mount on the host that shares a pod's lifetime
	NFS *NFSVolumeSource `json:"nfs,omitempty" description:"NFS volume that will be mounted in the host machine; see http://releases.k8s.io/v1.0.0/docs/volumes.md#nfs"`
	// ISCSI represents an ISCSI Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	ISCSI *ISCSIVolumeSource `json:"iscsi,omitempty" description:"iSCSI disk attached to host machine on demand; see http://releases.k8s.io/v1.0.0/examples/iscsi/README.md"`
	// Glusterfs represents a Glusterfs mount on the host that shares a pod's lifetime
	Glusterfs *GlusterfsVolumeSource `json:"glusterfs,omitempty" description:"Glusterfs volume that will be mounted on the host machine; see http://releases.k8s.io/v1.0.0/examples/glusterfs/README.md"`
	// PersistentVolumeClaimVolumeSource represents a reference to a PersistentVolumeClaim in the same namespace
	PersistentVolumeClaim *PersistentVolumeClaimVolumeSource `json:"persistentVolumeClaim,omitempty" description:"a reference to a PersistentVolumeClaim in the same namespace; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#persistentvolumeclaims"`
	// RBD represents a Rados Block Device mount on the host that shares a pod's lifetime
	RBD *RBDVolumeSource `json:"rbd,omitempty" description:"rados block volume that will be mounted on the host machine; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md"`
}

type PersistentVolumeClaimVolumeSource struct {
	// ClaimName is the name of a PersistentVolumeClaim in the same namespace as the pod using this volume
	ClaimName string `json:"claimName" description:"the name of the claim in the same namespace to be mounted as a volume; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#persistentvolumeclaims"`
	// Optional: Defaults to false (read/write).  ReadOnly here
	// will force the ReadOnly setting in VolumeMounts
	ReadOnly bool `json:"readOnly,omitempty" description:"mount volume as read-only when true; default false"`
}

// Similar to VolumeSource but meant for the administrator who creates PVs.
// Exactly one of its members must be set.
type PersistentVolumeSource struct {
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDiskVolumeSource `json:"gcePersistentDisk,omitempty" description:"GCE disk resource provisioned by an admin; see http://releases.k8s.io/v1.0.0/docs/volumes.md#gcepersistentdisk"`
	// AWSElasticBlockStore represents an AWS Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	AWSElasticBlockStore *AWSElasticBlockStoreVolumeSource `json:"awsElasticBlockStore,omitempty" description:"AWS disk resource provisioned by an admin; see http://releases.k8s.io/v1.0.0/docs/volumes.md#awselasticblockstore"`
	// HostPath represents a directory on the host.
	// This is useful for development and testing only.
	// on-host storage is not supported in any way.
	HostPath *HostPathVolumeSource `json:"hostPath,omitempty" description:"a HostPath provisioned by a developer or tester; for develment use only; see http://releases.k8s.io/v1.0.0/docs/volumes.md#hostpath"`
	// Glusterfs represents a Glusterfs volume that is attached to a host and exposed to the pod
	Glusterfs *GlusterfsVolumeSource `json:"glusterfs,omitempty" description:"Glusterfs volume resource provisioned by an admin; see http://releases.k8s.io/v1.0.0/examples/glusterfs/README.md"`
	// NFS represents an NFS mount on the host
	NFS *NFSVolumeSource `json:"nfs,omitempty" description:"NFS volume resource provisioned by an admin; see http://releases.k8s.io/v1.0.0/docs/volumes.md#nfs"`
	// RBD represents a Rados Block Device mount on the host that shares a pod's lifetime
	RBD *RBDVolumeSource `json:"rbd,omitempty" description:"rados block volume that will be mounted on the host machine; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md"`
	// ISCSI represents an ISCSI Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	ISCSI *ISCSIVolumeSource `json:"iscsi,omitempty" description:"an iSCSI disk resource provisioned by an admin"`
}

type PersistentVolume struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines a persistent volume owned by the cluster
	Spec PersistentVolumeSpec `json:"spec,omitempty" description:"specification of a persistent volume as provisioned by an administrator; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#persistent-volumes"`

	// Status represents the current information about persistent volume.
	Status PersistentVolumeStatus `json:"status,omitempty" description:"current status of a persistent volume; populated by the system, read-only; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#persistent-volumes"`
}

type PersistentVolumeSpec struct {
	// Resources represents the actual resources of the volume
	Capacity ResourceList `json:"capacity,omitempty" description:"a description of the persistent volume's resources and capacityr; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#capacity"`
	// Source represents the location and type of a volume to mount.
	PersistentVolumeSource `json:",inline" description:"the actual volume backing the persistent volume"`
	// AccessModes contains all ways the volume can be mounted
	AccessModes []PersistentVolumeAccessMode `json:"accessModes,omitempty" description:"all ways the volume can be mounted; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#access-modes"`
	// ClaimRef is part of a bi-directional binding between PersistentVolume and PersistentVolumeClaim.
	// ClaimRef is expected to be non-nil when bound.
	// claim.VolumeName is the authoritative bind between PV and PVC.
	ClaimRef *ObjectReference `json:"claimRef,omitempty" description:"when bound, a reference to the bound claim; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#binding"`
	// Optional: what happens to a persistent volume when released from its claim.
	PersistentVolumeReclaimPolicy PersistentVolumeReclaimPolicy `json:"persistentVolumeReclaimPolicy,omitempty" description:"what happens to a volume when released from its claim; Valid options are Retain (default) and Recycle.  Recyling must be supported by the volume plugin underlying this persistent volume. See http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#recycling-policy"`
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
	Phase PersistentVolumePhase `json:"phase,omitempty" description:"the current phase of a persistent volume; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#phase"`
	// A human-readable message indicating details about why the volume is in this state.
	Message string `json:"message,omitempty" description:"human-readable message indicating details about why the volume is in this state"`
	// Reason is a brief CamelCase string that describes any failure and is meant for machine parsing and tidy display in the CLI
	Reason string `json:"reason,omitempty" description:"(brief) reason the volume is not is not available"`
}

type PersistentVolumeList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#types-kinds"`
	Items    []PersistentVolume `json:"items,omitempty" description:"list of persistent volumes; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md"`
}

// PersistentVolumeClaim is a user's request for and claim to a persistent volume
type PersistentVolumeClaim struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the volume requested by a pod author
	Spec PersistentVolumeClaimSpec `json:"spec,omitempty" description:"the desired characteristics of a volume; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#persistentvolumeclaims"`

	// Status represents the current information about a claim
	Status PersistentVolumeClaimStatus `json:"status,omitempty" description:"the current status of a persistent volume claim; read-only; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#persistentvolumeclaims"`
}

type PersistentVolumeClaimList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#types-kinds"`
	Items    []PersistentVolumeClaim `json:"items,omitempty" description:"a list of persistent volume claims; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#persistentvolumeclaims"`
}

// PersistentVolumeClaimSpec describes the common attributes of storage devices
// and allows a Source for provider-specific attributes
type PersistentVolumeClaimSpec struct {
	// Contains the types of access modes required
	AccessModes []PersistentVolumeAccessMode `json:"accessModes,omitempty" description:"the desired access modes the volume should have; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#access-modes-1"`
	// Resources represents the minimum resources required
	Resources ResourceRequirements `json:"resources,omitempty" description:"the desired resources the volume should have; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#resources"`
	// VolumeName is the binding reference to the PersistentVolume backing this claim
	VolumeName string `json:"volumeName,omitempty" description:"the binding reference to the persistent volume backing this claim"`
}

type PersistentVolumeClaimStatus struct {
	// Phase represents the current phase of PersistentVolumeClaim
	Phase PersistentVolumeClaimPhase `json:"phase,omitempty" description:"the current phase of the claim"`
	// AccessModes contains all ways the volume backing the PVC can be mounted
	AccessModes []PersistentVolumeAccessMode `json:"accessModes,omitempty" description:"the actual access modes the volume has; see http://releases.k8s.io/v1.0.0/docs/persistent-volumes.md#access-modes-1"`
	// Represents the actual resources of the underlying volume
	Capacity ResourceList `json:"capacity,omitempty" description:"the actual resources the volume has"`
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

// HostPathVolumeSource represents bare host directory volume.
type HostPathVolumeSource struct {
	Path string `json:"path" description:"path of the directory on the host; see http://releases.k8s.io/v1.0.0/docs/volumes.md#hostpath"`
}

type EmptyDirVolumeSource struct {
	// Optional: what type of storage medium should back this directory.
	// The default is "" which means to use the node's default medium.
	Medium StorageMedium `json:"medium,omitempty" description:"type of storage used to back the volume; must be an empty string (default) or Memory; see http://releases.k8s.io/v1.0.0/docs/volumes.md#emptydir"`
}

// GlusterfsVolumeSource represents a Glusterfs Mount that lasts the lifetime of a pod
type GlusterfsVolumeSource struct {
	// Required: EndpointsName is the endpoint name that details Glusterfs topology
	EndpointsName string `json:"endpoints" description:"gluster hosts endpoints name; see http://releases.k8s.io/v1.0.0/examples/glusterfs/README.md#create-a-pod"`

	// Required: Path is the Glusterfs volume path
	Path string `json:"path" description:"path to gluster volume; see http://releases.k8s.io/v1.0.0/examples/glusterfs/README.md#create-a-pod"`

	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the Glusterfs volume to be mounted with read-only permissions
	ReadOnly bool `json:"readOnly,omitempty" description:"glusterfs volume to be mounted with read-only permissions; see http://releases.k8s.io/v1.0.0/examples/glusterfs/README.md#create-a-pod"`
}

// StorageMedium defines ways that storage can be allocated to a volume.
type StorageMedium string

// RBDVolumeSource represents a Rados Block Device Mount that lasts the lifetime of a pod
type RBDVolumeSource struct {
	// Required: CephMonitors is a collection of Ceph monitors
	CephMonitors []string `json:"monitors" description:"a collection of Ceph monitors; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
	// Required: RBDImage is the rados image name
	RBDImage string `json:"image" description:"rados image name; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType,omitempty" description:"file system type to mount, such as ext4, xfs, ntfs; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
	// Optional: RadosPool is the rados pool name,default is rbd
	RBDPool string `json:"pool" description:"rados pool name; default is rbd; optional; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
	// Optional: RBDUser is the rados user name, default is admin
	RadosUser string `json:"user" description:"rados user name; default is admin; optional; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
	// Optional: Keyring is the path to key ring for RBDUser, default is /etc/ceph/keyring
	Keyring string `json:"keyring" description:"keyring is the path to key ring for rados user; default is /etc/ceph/keyring; optional; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
	// Optional: SecretRef is name of the authentication secret for RBDUser, default is empty.
	SecretRef *LocalObjectReference `json:"secretRef" description:"name of a secret to authenticate the RBD user; if provided overrides keyring; optional; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty"  description:"rbd volume to be mounted with read-only permissions; see http://releases.k8s.io/v1.0.0/examples/rbd/README.md#how-to-use-it"`
}

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
	PDName string `json:"pdName" description:"unique name of the PD resource in GCE; see http://releases.k8s.io/v1.0.0/docs/volumes.md#gcepersistentdisk"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType" description:"file system type to mount, such as ext4, xfs, ntfs; see http://releases.k8s.io/v1.0.0/docs/volumes.md#gcepersistentdisk"`
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field is 0 or empty.
	Partition int `json:"partition,omitempty" description:"partition on the disk to mount (e.g., '1' for /dev/sda1); if omitted the plain device name (e.g., /dev/sda) will be mounted; see http://releases.k8s.io/v1.0.0/docs/volumes.md#gcepersistentdisk"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty" description:"read-only if true, read-write otherwise (false or unspecified); see http://releases.k8s.io/v1.0.0/docs/volumes.md#gcepersistentdisk"`
}

// AWSElasticBlockStoreVolumeSource represents a Persistent Disk resource in AWS.
//
// An AWS PD must exist and be formatted before mounting to a container.
// The disk must also be in the same AWS zone as the kubelet.
// A AWS PD can only be mounted on a single machine.
type AWSElasticBlockStoreVolumeSource struct {
	// Unique id of the PD resource. Used to identify the disk in AWS
	VolumeID string `json:"volumeID" description:"unique id of the PD resource in AWS; see http://releases.k8s.io/v1.0.0/docs/volumes.md#awselasticblockstore"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType" description:"file system type to mount, such as ext4, xfs, ntfs; see http://releases.k8s.io/v1.0.0/docs/volumes.md#awselasticblockstore"`
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field 0 or empty.
	Partition int `json:"partition,omitempty" description:"partition on the disk to mount (e.g., '1' for /dev/sda1); if omitted the plain device name (e.g., /dev/sda) will be mounted; see http://releases.k8s.io/v1.0.0/docs/volumes.md#awselasticblockstore"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty" description:"read-only if true, read-write otherwise (false or unspecified); see http://releases.k8s.io/v1.0.0/docs/volumes.md#awselasticblockstore"`
}

// GitRepoVolumeSource represents a volume that is pulled from git when the pod is created.
type GitRepoVolumeSource struct {
	// Repository URL
	Repository string `json:"repository" description:"repository URL"`
	// Commit hash, this is optional
	Revision string `json:"revision" description:"commit hash for the specified revision"`
}

// SecretVolumeSource adapts a Secret into a VolumeSource
//
// http://releases.k8s.io/v1.0.0/docs/design/secrets.md
type SecretVolumeSource struct {
	// Name of the secret in the pod's namespace to use
	SecretName string `json:"secretName" description:"secretName is the name of a secret in the pod's namespace; see http://releases.k8s.io/v1.0.0/docs/volumes.md#secrets"`
}

// NFSVolumeSource represents an NFS mount that lasts the lifetime of a pod
type NFSVolumeSource struct {
	// Server is the hostname or IP address of the NFS server
	Server string `json:"server" description:"the hostname or IP address of the NFS server; see http://releases.k8s.io/v1.0.0/docs/volumes.md#nfs"`

	// Path is the exported NFS share
	Path string `json:"path" description:"the path that is exported by the NFS server; see http://releases.k8s.io/v1.0.0/docs/volumes.md#nfs"`

	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the NFS export to be mounted with read-only permissions
	ReadOnly bool `json:"readOnly,omitempty" description:"forces the NFS export to be mounted with read-only permissions; see http://releases.k8s.io/v1.0.0/docs/volumes.md#nfs"`
}

// A ISCSI Disk can only be mounted as read/write once.
type ISCSIVolumeSource struct {
	// Required: iSCSI target portal
	// the portal is either an IP or ip_addr:port if port is other than default (typically TCP ports 860 and 3260)
	TargetPortal string `json:"targetPortal" description:"iSCSI target portal"`
	// Required:  target iSCSI Qualified Name
	IQN string `json:"iqn" description:"iSCSI Qualified Name"`
	// Required: iSCSI target lun number
	Lun int `json:"lun" description:"iscsi target lun number"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `json:"fsType" description:"file system type to mount, such as ext4, xfs, ntfs"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty" description:"read-only if true, read-write otherwise (false or unspecified)"`
}

// ContainerPort represents a network port in a single container.
type ContainerPort struct {
	// Optional: If specified, this must be an IANA_SVC_NAME  Each named port
	// in a pod must have a unique name.
	Name string `json:"name,omitempty" description:"name for the port that can be referred to by services; must be an IANA_SVC_NAME and unique within the pod"`
	// Optional: If specified, this must be a valid port number, 0 < x < 65536.
	// If HostNetwork is specified, this must match ContainerPort.
	HostPort int `json:"hostPort,omitempty" description:"number of port to expose on the host; most containers do not need this"`
	// Required: This must be a valid port number, 0 < x < 65536.
	ContainerPort int `json:"containerPort" description:"number of port to expose on the pod's IP address"`
	// Optional: Defaults to "TCP".
	Protocol Protocol `json:"protocol,omitempty" description:"protocol for port; must be UDP or TCP; TCP if unspecified"`
	// Optional: What host IP to bind the external port to.
	HostIP string `json:"hostIP,omitempty" description:"host IP to bind the port to"`
}

// VolumeMount describes a mounting of a Volume within a container.
type VolumeMount struct {
	// Required: This must match the Name of a Volume [above].
	Name string `json:"name" description:"name of the volume to mount"`
	// Optional: Defaults to false (read-write).
	ReadOnly bool `json:"readOnly,omitempty" description:"mounted read-only if true, read-write otherwise (false or unspecified)"`
	// Required.
	MountPath string `json:"mountPath" description:"path within the container at which the volume should be mounted"`
}

// EnvVar represents an environment variable present in a Container.
type EnvVar struct {
	// Required: This must be a C_IDENTIFIER.
	Name string `json:"name" description:"name of the environment variable; must be a C_IDENTIFIER"`
	// Optional: no more than one of the following may be specified.
	// Optional: Defaults to ""; variable references $(VAR_NAME) are expanded
	// using the previous defined environment variables in the container and
	// any service environment variables.  If a variable cannot be resolved,
	// the reference in the input string will be unchanged.  The $(VAR_NAME)
	// syntax can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped
	// references will never be expanded, regardless of whether the variable
	// exists or not.
	Value string `json:"value,omitempty" description:"value of the environment variable; defaults to empty string; variable references $(VAR_NAME) are expanded using the previously defined environment varibles in the container and any service environment variables; if a variable cannot be resolved, the reference in the input string will be unchanged; the $(VAR_NAME) syntax can be escaped with a double $$, ie: $$(VAR_NAME) ; escaped references will never be expanded, regardless of whether the variable exists or not"`
	// Optional: Specifies a source the value of this var should come from.
	ValueFrom *EnvVarSource `json:"valueFrom,omitempty" description:"source for the environment variable's value; cannot be used if value is not empty"`
}

// EnvVarSource represents a source for the value of an EnvVar.
type EnvVarSource struct {
	// Required: Selects a field of the pod; only name and namespace are supported.
	FieldRef *ObjectFieldSelector `json:"fieldRef" description:"selects a field of the pod; only name and namespace are supported"`
}

// ObjectFieldSelector selects an APIVersioned field of an object.
type ObjectFieldSelector struct {
	// Optional: Version of the schema the FieldPath is written in terms of, defaults to "v1"
	APIVersion string `json:"apiVersion,omitempty" description:"version of the schema that fieldPath is written in terms of; defaults to v1"`
	// Required: Path of the field to select in the specified API version
	FieldPath string `json:"fieldPath" description:"path of the field to select in the specified API version"`
}

// HTTPGetAction describes an action based on HTTP Get requests.
type HTTPGetAction struct {
	// Optional: Path to access on the HTTP server.
	Path string `json:"path,omitempty" description:"path to access on the HTTP server"`
	// Required: Name or number of the port to access on the container.
	Port util.IntOrString `json:"port" description:"number or name of the port to access on the container; number must be in the range 1 to 65535; name must be an IANA_SVC_NAME"`
	// Optional: Host name to connect to, defaults to the pod IP.
	Host string `json:"host,omitempty" description:"hostname to connect to; defaults to pod IP"`
	// Optional: Scheme to use for connecting to the host, defaults to HTTP.
	Scheme URIScheme `json:"scheme,omitempty" description:"scheme to connect with, must be HTTP or HTTPS, defaults to HTTP"`
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
	Port util.IntOrString `json:"port" description:"number of name of the port to access on the container; number must be in the range 1 to 65535; name must be an IANA_SVC_NAME"`
}

// ExecAction describes a "run in container" action.
type ExecAction struct {
	// Command is the command line to execute inside the container, the working directory for the
	// command  is root ('/') in the container's filesystem.  The command is simply exec'd, it is
	// not run inside a shell, so traditional shell instructions ('|', etc) won't work.  To use
	// a shell, you need to explicitly call out to that shell.
	Command []string `json:"command,omitempty" description:"command line to execute inside the container; working directory for the command is root ('/') in the container's file system; the command is exec'd, not run inside a shell; exit status of 0 is treated as live/healthy and non-zero is unhealthy"`
}

// Probe describes a liveness probe to be examined to the container.
type Probe struct {
	// The action taken to determine the health of a container
	Handler `json:",inline"`
	// Length of time before health checking is activated.  In seconds.
	InitialDelaySeconds int64 `json:"initialDelaySeconds,omitempty" description:"number of seconds after the container has started before liveness probes are initiated; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#container-probes"`
	// Length of time before health checking times out.  In seconds.
	TimeoutSeconds int64 `json:"timeoutSeconds,omitempty" description:"number of seconds after which liveness probes timeout; defaults to 1 second; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#container-probes"`
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
	Add []Capability `json:"add,omitempty" description:"added capabilities"`
	// Removed capabilities
	Drop []Capability `json:"drop,omitempty" description:"droped capabilities"`
}

// ResourceRequirements describes the compute resource requirements.
type ResourceRequirements struct {
	// Limits describes the maximum amount of compute resources required.
	Limits ResourceList `json:"limits,omitempty" description:"Maximum amount of compute resources allowed; see http://releases.k8s.io/v1.0.0/docs/design/resources.md#resource-specifications"`
	// Requests describes the minimum amount of compute resources required.
	// Note: 'Requests' are honored only for Persistent Volumes as of now.
	// TODO: Update the scheduler to use 'Requests' in addition to 'Limits'. If Request is omitted for a container,
	// it defaults to Limits if that is explicitly specified, otherwise to an implementation-defined value
	Requests ResourceList `json:"requests,omitempty" description:"Minimum amount of resources requested; requests are honored only for persistent volumes as of now; see http://releases.k8s.io/v1.0.0/docs/design/resources.md#resource-specifications"`
}

const (
	// TerminationMessagePathDefault means the default path to capture the application termination message running in a container
	TerminationMessagePathDefault string = "/dev/termination-log"
)

// Container represents a single container that is expected to be run on the host.
type Container struct {
	// Required: This must be a DNS_LABEL.  Each container in a pod must
	// have a unique name.
	Name string `json:"name" description:"name of the container; must be a DNS_LABEL and unique within the pod; cannot be updated"`
	// Optional.
	Image string `json:"image,omitempty" description:"Docker image name; see http://releases.k8s.io/v1.0.0/docs/images.md"`
	// Optional: The docker image's entrypoint is used if this is not provided; cannot be updated.
	// Variable references $(VAR_NAME) are expanded using the container's environment.  If a variable
	// cannot be resolved, the reference in the input string will be unchanged.  The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	Command []string `json:"command,omitempty" description:"entrypoint array; not executed within a shell; the docker image's entrypoint is used if this is not provided; cannot be updated; variable references $(VAR_NAME) are expanded using the container's environment variables; if a variable cannot be resolved, the reference in the input string will be unchanged; the $(VAR_NAME) syntax can be escaped with a double $$, ie: $$(VAR_NAME) ; escaped references will never be expanded, regardless of whether the variable exists or not; see http://releases.k8s.io/v1.0.0/docs/containers.md#containers-and-commands"`
	// Optional: The docker image's cmd is used if this is not provided; cannot be updated.
	// Variable references $(VAR_NAME) are expanded using the container's environment.  If a variable
	// cannot be resolved, the reference in the input string will be unchanged.  The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME).  Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	Args []string `json:"args,omitempty" description:"command array; the docker image's cmd is used if this is not provided; arguments to the entrypoint; cannot be updated; variable references $(VAR_NAME) are expanded using the container's environment variables; if a variable cannot be resolved, the reference in the input string will be unchanged; the $(VAR_NAME) syntax can be escaped with a double $$, ie: $$(VAR_NAME) ; escaped references will never be expanded, regardless of whether the variable exists or not; see http://releases.k8s.io/v1.0.0/docs/containers.md#containers-and-commands"`
	// Optional: Defaults to Docker's default.
	WorkingDir     string               `json:"workingDir,omitempty" description:"container's working directory; defaults to image's default; cannot be updated"`
	Ports          []ContainerPort      `json:"ports,omitempty" description:"list of ports to expose from the container; cannot be updated" patchStrategy:"merge" patchMergeKey:"containerPort"`
	Env            []EnvVar             `json:"env,omitempty" description:"list of environment variables to set in the container; cannot be updated" patchStrategy:"merge" patchMergeKey:"name"`
	Resources      ResourceRequirements `json:"resources,omitempty" description:"Compute Resources required by this container; cannot be updated; see http://releases.k8s.io/v1.0.0/docs/compute_resources.md"`
	VolumeMounts   []VolumeMount        `json:"volumeMounts,omitempty" description:"pod volumes to mount into the container's filesyste; cannot be updated" patchStrategy:"merge" patchMergeKey:"name"`
	LivenessProbe  *Probe               `json:"livenessProbe,omitempty" description:"periodic probe of container liveness; container will be restarted if the probe fails; cannot be updated; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#container-probes"`
	ReadinessProbe *Probe               `json:"readinessProbe,omitempty" description:"periodic probe of container service readiness; container will be removed from service endpoints if the probe fails; cannot be updated; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#container-probes"`
	Lifecycle      *Lifecycle           `json:"lifecycle,omitempty" description:"actions that the management system should take in response to container lifecycle events; cannot be updated"`
	// Optional: Defaults to /dev/termination-log
	TerminationMessagePath string `json:"terminationMessagePath,omitempty" description:"path at which the file to which the container's termination message will be written is mounted into the container's filesystem; message written is intended to be brief final status, such as an assertion failure message; defaults to /dev/termination-log; cannot be updated"`
	// Optional: Policy for pulling images for this container
	ImagePullPolicy PullPolicy `json:"imagePullPolicy,omitempty" description:"image pull policy; one of Always, Never, IfNotPresent; defaults to Always if :latest tag is specified, or IfNotPresent otherwise; cannot be updated; see http://releases.k8s.io/v1.0.0/docs/images.md#updating-images"`
	// Optional: SecurityContext defines the security options the pod should be run with
	SecurityContext *SecurityContext `json:"securityContext,omitempty" description:"security options the pod should run with; see http://releases.k8s.io/v1.0.0/docs/security_context.md"`
}

// Handler defines a specific action that should be taken
// TODO: pass structured data to these actions, and document that data here.
type Handler struct {
	// One and only one of the following should be specified.
	// Exec specifies the action to take.
	Exec *ExecAction `json:"exec,omitempty" description:"exec-based handler"`
	// HTTPGet specifies the http request to perform.
	HTTPGet *HTTPGetAction `json:"httpGet,omitempty" description:"HTTP-based handler"`
	// TCPSocket specifies an action involving a TCP port.
	// TODO: implement a realistic TCP lifecycle hook
	TCPSocket *TCPSocketAction `json:"tcpSocket,omitempty"  description:"TCP-based handler; TCP hooks not yet supported"`
}

// Lifecycle describes actions that the management system should take in response to container lifecycle
// events.  For the PostStart and PreStop lifecycle handlers, management of the container blocks
// until the action is complete, unless the container process fails, in which case the handler is aborted.
type Lifecycle struct {
	// PostStart is called immediately after a container is created.  If the handler fails, the container
	// is terminated and restarted.
	PostStart *Handler `json:"postStart,omitempty" description:"called immediately after a container is started; if the handler fails, the container is terminated and restarted according to its restart policy; other management of the container blocks until the hook completes; see http://releases.k8s.io/v1.0.0/docs/container-environment.md#hook-details"`
	// PreStop is called immediately before a container is terminated.  The reason for termination is
	// passed to the handler.  Regardless of the outcome of the handler, the container is eventually terminated.
	PreStop *Handler `json:"preStop,omitempty" description:"called before a container is terminated; the container is terminated after the handler completes; other management of the container blocks until the hook completes; see http://releases.k8s.io/v1.0.0/docs/container-environment.md#hook-details"`
}

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
	Reason string `json:"reason,omitempty" description:"(brief) reason the container is not yet running, such as pulling its image"`
}

type ContainerStateRunning struct {
	StartedAt util.Time `json:"startedAt,omitempty" description:"time at which the container was last (re-)started"`
}

type ContainerStateTerminated struct {
	ExitCode    int       `json:"exitCode" description:"exit status from the last termination of the container"`
	Signal      int       `json:"signal,omitempty" description:"signal from the last termination of the container"`
	Reason      string    `json:"reason,omitempty" description:"(brief) reason from the last termination of the container"`
	Message     string    `json:"message,omitempty" description:"message regarding the last termination of the container"`
	StartedAt   util.Time `json:"startedAt,omitempty" description:"time at which previous execution of the container started"`
	FinishedAt  util.Time `json:"finishedAt,omitempty" description:"time at which the container last terminated"`
	ContainerID string    `json:"containerID,omitempty" description:"container's ID in the format 'docker://<container_id>'"`
}

// ContainerState holds a possible state of container.
// Only one of its members may be specified.
// If none of them is specified, the default one is ContainerStateWaiting.
type ContainerState struct {
	Waiting    *ContainerStateWaiting    `json:"waiting,omitempty" description:"details about a waiting container"`
	Running    *ContainerStateRunning    `json:"running,omitempty" description:"details about a running container"`
	Terminated *ContainerStateTerminated `json:"terminated,omitempty" description:"details about a terminated container"`
}

type ContainerStatus struct {
	// Required: This must be a DNS_LABEL.  Each container in a pod must have a unique name.
	Name string `json:"name" description:"name of the container; must be a DNS_LABEL and unique within the pod; cannot be updated"`
	// TODO(dchen1107): Should we rename PodStatus to a more generic name or have a separate states
	// defined for container?
	State                ContainerState `json:"state,omitempty" description:"details about the container's current condition"`
	LastTerminationState ContainerState `json:"lastState,omitempty" description:"details about the container's last termination condition"`
	Ready                bool           `json:"ready" description:"specifies whether the container has passed its readiness probe"`
	// Note that this is calculated from dead containers.  But those containers are subject to
	// garbage collection.  This value will get capped at 5 by GC.
	RestartCount int `json:"restartCount" description:"the number of times the container has been restarted, currently based on the number of dead containers that have not yet been removed"`
	// TODO(dchen1107): Which image the container is running with?
	// The image the container is running
	Image       string `json:"image" description:"image of the container; see http://releases.k8s.io/v1.0.0/docs/images.md"`
	ImageID     string `json:"imageID" description:"ID of the container's image"`
	ContainerID string `json:"containerID,omitempty" description:"container's ID in the format 'docker://<container_id>'; see http://releases.k8s.io/v1.0.0/docs/container-environment.md#container-information"`
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

// PodConditionType is a valid value for PodCondition.Type
type PodConditionType string

// These are valid conditions of pod.
const (
	// PodReady means the pod is able to service requests and should be added to the
	// load balancing pools of all matching services.
	PodReady PodConditionType = "Ready"
)

// TODO: add LastTransitionTime, Reason, Message to match NodeCondition api.
type PodCondition struct {
	// Type is the type of the condition
	Type PodConditionType `json:"type" description:"kind of the condition, currently only Ready; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#pod-conditions"`
	// Status is the status of the condition
	Status ConditionStatus `json:"status" description:"status of the condition, one of True, False, Unknown; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#pod-conditions"`
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
	Volumes []Volume `json:"volumes,omitempty" description:"list of volumes that can be mounted by containers belonging to the pod; see http://releases.k8s.io/v1.0.0/docs/volumes.md" patchStrategy:"merge" patchMergeKey:"name"`
	// Required: there must be at least one container in a pod.
	Containers    []Container   `json:"containers" description:"list of containers belonging to the pod; cannot be updated; containers cannot currently be added or removed; there must be at least one container in a Pod; see http://releases.k8s.io/v1.0.0/docs/containers.md" patchStrategy:"merge" patchMergeKey:"name"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty" description:"restart policy for all containers within the pod; one of Always, OnFailure, Never; defaults to Always; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#restartpolicy"`
	// Optional duration in seconds the pod needs to terminate gracefully. May be decreased in delete request.
	// Value must be non-negative integer. The value zero indicates delete immediately.
	// If this value is nil, the default grace period will be used instead.
	// The grace period is the duration in seconds after the processes running in the pod are sent
	// a termination signal and the time when the processes are forcibly halted with a kill signal.
	// Set this value longer than the expected cleanup time for your process.
	TerminationGracePeriodSeconds *int64 `json:"terminationGracePeriodSeconds,omitempty" description:"optional duration in seconds the pod needs to terminate gracefully; may be decreased in delete request; value must be non-negative integer; the value zero indicates delete immediately; if this value is not set, the default grace period will be used instead; the grace period is the duration in seconds after the processes running in the pod are sent a termination signal and the time when the processes are forcibly halted with a kill signal; set this value longer than the expected cleanup time for your process"`
	ActiveDeadlineSeconds         *int64 `json:"activeDeadlineSeconds,omitempty" description:"optional duration in seconds the pod may be active on the node relative to StartTime before the system will actively try to mark it failed and kill associated containers; value must be a positive integer`
	// Optional: Set DNS policy.  Defaults to "ClusterFirst"
	DNSPolicy DNSPolicy `json:"dnsPolicy,omitempty" description:"DNS policy for containers within the pod; one of 'ClusterFirst' or 'Default'"`
	// NodeSelector is a selector which must be true for the pod to fit on a node
	NodeSelector map[string]string `json:"nodeSelector,omitempty" description:"selector which must match a node's labels for the pod to be scheduled on that node; see http://releases.k8s.io/v1.0.0/examples/node-selection/README.md"`

	// ServiceAccountName is the name of the ServiceAccount to use to run this pod
	ServiceAccountName string `json:"serviceAccountName,omitempty" description:"name of the ServiceAccount to use to run this pod; see http://releases.k8s.io/v1.0.0/docs/service_accounts.md"`

	// NodeName is a request to schedule this pod onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this pod onto that node, assuming that it fits resource
	// requirements.
	NodeName string `json:"nodeName,omitempty" description:"node requested for this pod"`
	// Uses the host's network namespace. If this option is set, the ports that will be
	// used must be specified.
	// Optional: Default to false.
	HostNetwork bool `json:"hostNetwork,omitempty" description:"host networking requested for this pod"`
	// ImagePullSecrets is an optional list of references to secrets in the same namespace to use for pulling any of the images used by this PodSpec.
	// If specified, these secrets will be passed to individual puller implementations for them to use.  For example,
	// in the case of docker, only DockerConfig type secrets are honored.
	ImagePullSecrets []LocalObjectReference `json:"imagePullSecrets,omitempty" description:"list of references to secrets in the same namespace available for pulling the container images; see http://releases.k8s.io/v1.0.0/docs/images.md#specifying-imagepullsecrets-on-a-pod"  patchStrategy:"merge" patchMergeKey:"name"`
}

// PodStatus represents information about the status of a pod. Status may trail the actual
// state of a system.
type PodStatus struct {
	Phase      PodPhase       `json:"phase,omitempty" description:"current condition of the pod; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#pod-phase"`
	Conditions []PodCondition `json:"conditions,omitempty" description:"current service state of pod; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#pod-conditions" patchStrategy:"merge" patchMergeKey:"type"`
	// A human readable message indicating details about why the pod is in this state.
	Message string `json:"message,omitempty" description:"human readable message indicating details about why the pod is in this condition"`
	// A brief CamelCase message indicating details about why the pod is in this state. e.g. 'OutOfDisk'
	Reason string `json:"reason,omitempty" description:"(brief-CamelCase) reason indicating details about why the pod is in this condition"`

	HostIP string `json:"hostIP,omitempty" description:"IP address of the host to which the pod is assigned; empty if not yet scheduled"`
	PodIP  string `json:"podIP,omitempty" description:"IP address allocated to the pod; routable at least within the cluster; empty if not yet allocated"`

	StartTime *util.Time `json:"startTime,omitempty" description:"RFC 3339 date and time at which the object was acknowledged by the Kubelet.  This is before the Kubelet pulled the container image(s) for the pod."`

	// The list has one entry per container in the manifest. Each entry is currently the output
	// of `docker inspect`.
	ContainerStatuses []ContainerStatus `json:"containerStatuses,omitempty" description:"list of container statuses; see http://releases.k8s.io/v1.0.0/docs/pod-states.md#container-statuses"`
}

// PodStatusResult is a wrapper for PodStatus returned by kubelet that can be encode/decoded
type PodStatusResult struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`
	// Status represents the current information about a pod. This data may not be up
	// to date.
	Status PodStatus `json:"status,omitempty" description:"most recently observed status of the pod; populated by the system, read-only; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// Pod is a collection of containers that can run on a host. This resource is created
// by clients and scheduled onto hosts.
type Pod struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the behavior of a pod.
	Spec PodSpec `json:"spec,omitempty" description:"specification of the desired behavior of the pod; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`

	// Status represents the current information about a pod. This data may not be up
	// to date.
	Status PodStatus `json:"status,omitempty" description:"most recently observed status of the pod; populated by the system, read-only; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// PodList is a list of Pods.
type PodList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#types-kinds"`

	Items []Pod `json:"items" description:"list of pods; see http://releases.k8s.io/v1.0.0/docs/pods.md"`
}

// PodTemplateSpec describes the data a pod should have when created from a template
type PodTemplateSpec struct {
	// Metadata of the pods created from this template.
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the behavior of a pod.
	Spec PodSpec `json:"spec,omitempty" description:"specification of the desired behavior of the pod; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// PodTemplate describes a template for creating copies of a predefined pod.
type PodTemplate struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Template defines the pods that will be created from this pod template
	Template PodTemplateSpec `json:"template,omitempty" description:"the template of the desired behavior of the pod; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// PodTemplateList is a list of PodTemplates.
type PodTemplateList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []PodTemplate `json:"items" description:"list of pod templates"`
}

// ReplicationControllerSpec is the specification of a replication controller.
type ReplicationControllerSpec struct {
	// Replicas is the number of desired replicas. This is a pointer to distinguish between explicit zero and unspecified.
	Replicas *int `json:"replicas,omitempty" description:"number of replicas desired; defaults to 1; see http://releases.k8s.io/v1.0.0/docs/replication-controller.md#what-is-a-replication-controller"`

	// Selector is a label query over pods that should match the Replicas count.
	// If Selector is empty, it is defaulted to the labels present on the Pod template.
	Selector map[string]string `json:"selector,omitempty" description:"label keys and values that must match in order to be controlled by this replication controller, if empty defaulted to labels on Pod template; see http://releases.k8s.io/v1.0.0/docs/labels.md#label-selectors"`

	// TemplateRef is a reference to an object that describes the pod that will be created if
	// insufficient replicas are detected.
	//TemplateRef *ObjectReference `json:"templateRef,omitempty" description:"reference to an object that describes the pod that will be created if insufficient replicas are detected"`

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected. This takes precedence over a
	// TemplateRef.
	Template *PodTemplateSpec `json:"template,omitempty" description:"object that describes the pod that will be created if insufficient replicas are detected; takes precendence over templateRef; see http://releases.k8s.io/v1.0.0/docs/replication-controller.md#pod-template"`
}

// ReplicationControllerStatus represents the current status of a replication
// controller.
type ReplicationControllerStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int `json:"replicas" description:"most recently oberved number of replicas; see http://releases.k8s.io/v1.0.0/docs/replication-controller.md#what-is-a-replication-controller"`

	// ObservedGeneration is the most recent generation observed by the controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty" description:"reflects the generation of the most recently observed replication controller"`
}

// ReplicationController represents the configuration of a replication controller.
type ReplicationController struct {
	TypeMeta `json:",inline"`

	// If the Labels of a ReplicationController are empty, they are defaulted to be the same as the Pod(s) that the replication controller manages.
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the desired behavior of this replication controller.
	Spec ReplicationControllerSpec `json:"spec,omitempty" description:"specification of the desired behavior of the replication controller; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`

	// Status is the current status of this replication controller. This data may be
	// out of date by some window of time.
	Status ReplicationControllerStatus `json:"status,omitempty" description:"most recently observed status of the replication controller; populated by the system, read-only; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// ReplicationControllerList is a collection of replication controllers.
type ReplicationControllerList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []ReplicationController `json:"items" description:"list of replication controllers; see http://releases.k8s.io/v1.0.0/docs/replication-controller.md"`
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
	// cluster, via the cluster IP.
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
	LoadBalancer LoadBalancerStatus `json:"loadBalancer,omitempty" description:"status of load-balancer"`
}

// LoadBalancerStatus represents the status of a load-balancer
type LoadBalancerStatus struct {
	// Ingress is a list containing ingress points for the load-balancer;
	// traffic intended for the service should be sent to these ingress points.
	Ingress []LoadBalancerIngress `json:"ingress,omitempty" description:"load-balancer ingress points"`
}

// LoadBalancerIngress represents the status of a load-balancer ingress point:
// traffic intended for the service should be sent to an ingress point.
type LoadBalancerIngress struct {
	// IP is set for load-balancer ingress points that are IP based
	// (typically GCE or OpenStack load-balancers)
	IP string `json:"ip,omitempty" description:"IP address of ingress point"`

	// Hostname is set for load-balancer ingress points that are DNS based
	// (typically AWS load-balancers)
	Hostname string `json:"hostname,omitempty" description:"hostname of ingress point"`
}

// ServiceSpec describes the attributes that a user creates on a service
type ServiceSpec struct {
	// Required: The list of ports that are exposed by this service.
	Ports []ServicePort `json:"ports" description:"ports exposed by the service; see http://releases.k8s.io/v1.0.0/docs/services.md#virtual-ips-and-service-proxies"`

	// This service will route traffic to pods having labels matching this selector. If null, no endpoints will be automatically created. If empty, all pods will be selected.
	Selector map[string]string `json:"selector,omitempty" description:"label keys and values that must match in order to receive traffic for this service; if empty, all pods are selected, if not specified, endpoints must be manually specified; see http://releases.k8s.io/v1.0.0/docs/services.md#overview"`

	// ClusterIP is usually assigned by the master.  If specified by the user
	// we will try to respect it or else fail the request.  This field can
	// not be changed by updates.
	// Valid values are None, empty string (""), or a valid IP address
	// None can be specified for headless services when proxying is not required
	ClusterIP string `json:"clusterIP,omitempty" description:"IP address of the service; usually assigned by the system; if specified, it will be allocated to the service if unused or else creation of the service will fail; cannot be updated; 'None' can be specified for a headless service when proxying is not required; see http://releases.k8s.io/v1.0.0/docs/services.md#virtual-ips-and-service-proxies"`

	// Type determines how the service will be exposed.  Valid options: ClusterIP, NodePort, LoadBalancer
	Type ServiceType `json:"type,omitempty" description:"type of this service; must be ClusterIP, NodePort, or LoadBalancer; defaults to ClusterIP; see http://releases.k8s.io/v1.0.0/docs/services.md#external-services"`

	// Deprecated. PublicIPs are used by external load balancers, or can be set by
	// users to handle external traffic that arrives at a node.
	DeprecatedPublicIPs []string `json:"deprecatedPublicIPs,omitempty" description:"deprecated. externally visible IPs (e.g. load balancers) that should be proxied to this service"`

	// Optional: Supports "ClientIP" and "None".  Used to maintain session affinity.
	SessionAffinity ServiceAffinity `json:"sessionAffinity,omitempty" description:"enable client IP based session affinity; must be ClientIP or None; defaults to None; see http://releases.k8s.io/v1.0.0/docs/services.md#virtual-ips-and-service-proxies"`
}

type ServicePort struct {
	// Optional if only one ServicePort is defined on this service: The
	// name of this port within the service.  This must be a DNS_LABEL.
	// All ports within a ServiceSpec must have unique names.  This maps to
	// the 'Name' field in EndpointPort objects.
	Name string `json:"name,omitempty" description:"the name of this port; optional if only one port is defined"`

	// Optional: The IP protocol for this port.  Supports "TCP" and "UDP",
	// default is TCP.
	Protocol Protocol `json:"protocol,omitempty" description:"the protocol used by this port; must be UDP or TCP; TCP if unspecified"`

	// Required: The port that will be exposed by this service.
	Port int `json:"port" description:"the port number that is exposed"`

	// Optional: The target port on pods selected by this service.
	// If this is a string, it will be looked up as a named port in the
	// target Pod's container ports.  If this is not specified, the value
	// of Port is used (an identity map).
	TargetPort util.IntOrString `json:"targetPort,omitempty" description:"number or name of the port to access on the pods targeted by the service; defaults to the service port; number must be in the range 1 to 65535; name must be an IANA_SVC_NAME; see http://releases.k8s.io/v1.0.0/docs/services.md#defining-a-service"`

	// The port on each node on which this service is exposed.
	// Default is to auto-allocate a port if the ServiceType of this Service requires one.
	NodePort int `json:"nodePort" description:"the port on each node on which this service is exposed when type=NodePort or LoadBalancer; usually assigned by the system; if specified, it will be allocated to the service if unused or else creation of the service will fail; see http://releases.k8s.io/v1.0.0/docs/services.md#type--nodeport"`
}

// Service is a named abstraction of software service (for example, mysql) consisting of local port
// (for example 3306) that the proxy listens on, and the selector that determines which pods
// will answer requests sent through the proxy.
type Service struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the behavior of a service.
	Spec ServiceSpec `json:"spec,omitempty" description:"specification of the desired behavior of the service; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`

	// Status represents the current status of a service.
	Status ServiceStatus `json:"status,omitempty" description:"most recently observed status of the service; populated by the system, read-only; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

const (
	// ClusterIPNone - do not assign a cluster IP
	// no proxying required and no environment variables should be created for pods
	ClusterIPNone = "None"
)

// ServiceList holds a list of services.
type ServiceList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []Service `json:"items" description:"list of services"`
}

// ServiceAccount binds together:
// * a name, understood by users, and perhaps by peripheral systems, for an identity
// * a principal that can be authenticated and authorized
// * a set of secrets
type ServiceAccount struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Secrets is the list of secrets allowed to be used by pods running using this ServiceAccount
	Secrets []ObjectReference `json:"secrets,omitempty" description:"list of secrets that can be used by pods running as this service account; see http://releases.k8s.io/v1.0.0/docs/secrets.md" patchStrategy:"merge" patchMergeKey:"name"`

	// ImagePullSecrets is a list of references to secrets in the same namespace to use for pulling any images
	// in pods that reference this ServiceAccount.  ImagePullSecrets are distinct from Secrets because Secrets
	// can be mounted in the pod, but ImagePullSecrets are only accessed by the kubelet.
	ImagePullSecrets []LocalObjectReference `json:"imagePullSecrets,omitempty" description:"list of references to secrets in the same namespace available for pulling container images; see http://releases.k8s.io/v1.0.0/docs/secrets.md#manually-specifying-an-imagepullsecret"`
}

// ServiceAccountList is a list of ServiceAccount objects
type ServiceAccountList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []ServiceAccount `json:"items" description:"list of ServiceAccounts; see http://releases.k8s.io/v1.0.0/docs/service_accounts.md#service-accounts"`
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
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// The set of all endpoints is the union of all subsets.
	Subsets []EndpointSubset `json:"subsets" description:"sets of addresses and ports that comprise a service"`
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
	Addresses []EndpointAddress `json:"addresses,omitempty" description:"IP addresses which offer the related ports"`
	Ports     []EndpointPort    `json:"ports,omitempty" description:"port numbers available on the related IP addresses"`
}

// EndpointAddress is a tuple that describes single IP address.
type EndpointAddress struct {
	// The IP of this endpoint.
	// TODO: This should allow hostname or IP, see #4447.
	IP string `json:"ip" description:"IP address of the endpoint"`

	// Optional: The kubernetes object related to the entry point.
	TargetRef *ObjectReference `json:"targetRef,omitempty" description:"reference to object providing the endpoint"`
}

// EndpointPort is a tuple that describes a single port.
type EndpointPort struct {
	// The name of this port (corresponds to ServicePort.Name).  Optional
	// if only one port is defined.  Must be a DNS_LABEL.
	Name string `json:"name,omitempty" description:"name of this port"`

	// The port number.
	Port int `json:"port" description:"port number of the endpoint"`

	// The IP protocol for this port.
	Protocol Protocol `json:"protocol,omitempty" description:"protocol for this port; must be UDP or TCP; TCP if unspecified"`
}

// EndpointsList is a list of endpoints.
type EndpointsList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []Endpoints `json:"items" description:"list of endpoints"`
}

// NodeSpec describes the attributes that a node is created with.
type NodeSpec struct {
	// PodCIDR represents the pod IP range assigned to the node
	PodCIDR string `json:"podCIDR,omitempty" description:"pod IP range assigned to the node"`
	// External ID of the node assigned by some machine database (e.g. a cloud provider)
	ExternalID string `json:"externalID,omitempty" description:"deprecated. External ID assigned to the node by some machine database (e.g. a cloud provider). Defaults to node name when empty."`
	// ID of the node assigned by the cloud provider
	ProviderID string `json:"providerID,omitempty" description:"ID of the node assigned by the cloud provider in the format: <ProviderName>://<ProviderSpecificNodeID>"`
	// Unschedulable controls node schedulability of new pods. By default node is schedulable.
	Unschedulable bool `json:"unschedulable,omitempty" description:"disable pod scheduling on the node; see http://releases.k8s.io/v1.0.0/docs/node.md#manual-node-administration"`
}

// NodeSystemInfo is a set of ids/uuids to uniquely identify the node.
type NodeSystemInfo struct {
	// MachineID is the machine-id reported by the node
	MachineID string `json:"machineID" description:"machine-id reported by the node"`
	// SystemUUID is the system-uuid reported by the node
	SystemUUID string `json:"systemUUID" description:"system-uuid reported by the node"`
	// BootID is the boot-id reported by the node
	BootID string `json:"bootID" description:"boot id is the boot-id reported by the node"`
	// Kernel version reported by the node
	KernelVersion string `json:"kernelVersion" description:"Kernel version reported by the node from 'uname -r' (e.g. 3.16.0-0.bpo.4-amd64)"`
	// OS image used reported by the node
	OsImage string `json:"osImage" description:"OS image used reported by the node from /etc/os-release (e.g. Debian GNU/Linux 7 (wheezy))"`
	// Container runtime version reported by the node
	ContainerRuntimeVersion string `json:"containerRuntimeVersion" description:"Container runtime version reported by the node through runtime remote API (e.g. docker://1.5.0)"`
	// Kubelet version reported by the node
	KubeletVersion string `json:"kubeletVersion" description:"Kubelet version reported by the node"`
	// Kube-proxy version reported by the node
	KubeProxyVersion string `json:"kubeProxyVersion" description:"Kube-proxy version reported by the node"`
}

// NodeStatus is information about the current status of a node.
type NodeStatus struct {
	// Capacity represents the available resources of a node.
	// see http://releases.k8s.io/v1.0.0/docs/compute_resources.md for more details.
	Capacity ResourceList `json:"capacity,omitempty" description:"compute resource capacity of the node; see http://releases.k8s.io/v1.0.0/docs/compute_resources.md"`
	// NodePhase is the current lifecycle phase of the node.
	Phase NodePhase `json:"phase,omitempty" description:"most recently observed lifecycle phase of the node; see http://releases.k8s.io/v1.0.0/docs/node.md#node-phase"`
	// Conditions is an array of current node conditions.
	Conditions []NodeCondition `json:"conditions,omitempty" description:"list of node conditions observed; see http://releases.k8s.io/v1.0.0/docs/node.md#node-condition" patchStrategy:"merge" patchMergeKey:"type"`
	// Queried from cloud provider, if available.
	Addresses []NodeAddress `json:"addresses,omitempty" description:"list of addresses reachable to the node; see http://releases.k8s.io/v1.0.0/docs/node.md#node-addresses" patchStrategy:"merge" patchMergeKey:"type"`
	// NodeSystemInfo is a set of ids/uuids to uniquely identify the node
	NodeInfo NodeSystemInfo `json:"nodeInfo,omitempty" description:"set of ids/uuids to uniquely identify the node; see http://releases.k8s.io/v1.0.0/docs/node.md#node-info"`
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
// NodeReachable, NodeLive, NodeReady, NodeSchedulable, NodeRunnable.
const (
	// NodeReady means kubelet is healthy and ready to accept pods.
	NodeReady NodeConditionType = "Ready"
)

type NodeCondition struct {
	Type               NodeConditionType `json:"type" description:"type of node condition, currently only Ready"`
	Status             ConditionStatus   `json:"status" description:"status of the condition, one of True, False, Unknown"`
	LastHeartbeatTime  util.Time         `json:"lastHeartbeatTime,omitempty" description:"last time we got an update on a given condition"`
	LastTransitionTime util.Time         `json:"lastTransitionTime,omitempty" description:"last time the condition transit from one status to another"`
	Reason             string            `json:"reason,omitempty" description:"(brief) reason for the condition's last transition"`
	Message            string            `json:"message,omitempty" description:"human readable message indicating details about last transition"`
}

type NodeAddressType string

// These are valid address type of node.
const (
	NodeHostName   NodeAddressType = "Hostname"
	NodeExternalIP NodeAddressType = "ExternalIP"
	NodeInternalIP NodeAddressType = "InternalIP"
)

type NodeAddress struct {
	Type    NodeAddressType `json:"type" description:"node address type, one of Hostname, ExternalIP or InternalIP"`
	Address string          `json:"address" description:"the node address"`
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
)

// ResourceList is a set of (resource name, quantity) pairs.
type ResourceList map[ResourceName]resource.Quantity

// Node is a worker node in Kubernetes.
// The name of the node according to etcd is in ID.
type Node struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the behavior of a node.
	Spec NodeSpec `json:"spec,omitempty" description:"specification of a node; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`

	// Status describes the current status of a Node
	Status NodeStatus `json:"status,omitempty" description:"most recently observed status of the node; populated by the system, read-only; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// NodeList is a list of minions.
type NodeList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []Node `json:"items" description:"list of nodes"`
}

type FinalizerName string

// These are internal finalizer values to Kubernetes, must be qualified name unless defined here
const (
	FinalizerKubernetes FinalizerName = "kubernetes"
)

// NamespaceSpec describes the attributes on a Namespace
type NamespaceSpec struct {
	// Finalizers is an opaque list of values that must be empty to permanently remove object from storage
	Finalizers []FinalizerName `json:"finalizers,omitempty" description:"an opaque list of values that must be empty to permanently remove object from storage; see http://releases.k8s.io/v1.0.0/docs/design/namespaces.md#finalizers"`
}

// NamespaceStatus is information about the current status of a Namespace.
type NamespaceStatus struct {
	// Phase is the current lifecycle phase of the namespace.
	Phase NamespacePhase `json:"phase,omitempty" description:"phase is the current lifecycle phase of the namespace; see http://releases.k8s.io/v1.0.0/docs/design/namespaces.md#phases"`
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
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the behavior of the Namespace.
	Spec NamespaceSpec `json:"spec,omitempty" description:"spec defines the behavior of the Namespace; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`

	// Status describes the current status of a Namespace
	Status NamespaceStatus `json:"status,omitempty" description:"status describes the current status of a Namespace; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// NamespaceList is a list of Namespaces.
type NamespaceList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Items is the list of Namespace objects in the list
	Items []Namespace `json:"items"  description:"items is the list of Namespace objects in the list; see http://releases.k8s.io/v1.0.0/docs/namespaces.md"`
}

// Binding ties one object to another - for example, a pod is bound to a node by a scheduler.
type Binding struct {
	TypeMeta `json:",inline"`
	// ObjectMeta describes the object that is being bound.
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Target is the object to bind to.
	Target ObjectReference `json:"target" description:"an object to bind to"`
}

// DeleteOptions may be provided when deleting an API object
type DeleteOptions struct {
	TypeMeta `json:",inline"`

	// Optional duration in seconds before the object should be deleted. Value must be non-negative integer.
	// The value zero indicates delete immediately. If this value is nil, the default grace period for the
	// specified type will be used.
	GracePeriodSeconds *int64 `json:"gracePeriodSeconds" description:"the duration in seconds to wait before deleting this object; defaults to a per object value if not specified; zero means delete immediately"`
}

// ListOptions is the query options to a standard REST list call
type ListOptions struct {
	TypeMeta `json:",inline"`

	// A selector based on labels
	LabelSelector string `json:"labelSelector,omitempty" description:"a selector to restrict the list of returned objects by their labels; defaults to everything"`
	// A selector based on fields
	FieldSelector string `json:"fieldSelector,omitempty" description:"a selector to restrict the list of returned objects by their fields; defaults to everything"`
	// If true, watch for changes to the selected resources
	Watch bool `json:"watch,omitempty" description:"watch for changes to the described resources and return them as a stream of add, update, and remove notifications; specify resourceVersion"`
	// The desired resource version to watch
	ResourceVersion string `json:"resourceVersion,omitempty" description:"when specified with a watch call, shows changes that occur after that particular version of a resource; defaults to changes from the beginning of history"`
}

// PodLogOptions is the query options for a Pod's logs REST call
type PodLogOptions struct {
	TypeMeta `json:",inline"`

	// Container for which to return logs
	Container string `json:"container,omitempty" description:"the container for which to stream logs; defaults to only container if there is one container in the pod"`

	// If true, follow the logs for the pod
	Follow bool `json:"follow,omitempty" description:"follow the log stream of the pod; defaults to false"`

	//  If true, return previous terminated container logs
	Previous bool `json:"previous,omitempty" description:"return previous terminated container logs; defaults to false"`
}

// PodExecOptions is the query options to a Pod's remote exec call
type PodExecOptions struct {
	TypeMeta `json:",inline"`

	// Stdin if true indicates that stdin is to be redirected for the exec call
	Stdin bool `json:"stdin,omitempty" description:"redirect the standard input stream of the pod for this call; defaults to false"`

	// Stdout if true indicates that stdout is to be redirected for the exec call
	Stdout bool `json:"stdout,omitempty" description:"redirect the standard output stream of the pod for this call; defaults to true"`

	// Stderr if true indicates that stderr is to be redirected for the exec call
	Stderr bool `json:"stderr,omitempty" description:"redirect the standard error stream of the pod for this call; defaults to true"`

	// TTY if true indicates that a tty will be allocated for the exec call
	TTY bool `json:"tty,omitempty" description:"allocate a terminal for this exec call; defaults to false"`

	// Container in which to execute the command.
	Container string `json:"container,omitempty" description:"the container in which to execute the command. Defaults to only container if there is only one container in the pod."`

	// Command is the remote command to execute; argv array; not executed within a shell.
	Command []string `json:"command" description:"the command to execute; argv array; not executed within a shell"`
}

// PodProxyOptions is the query options to a Pod's proxy call
type PodProxyOptions struct {
	TypeMeta `json:",inline"`

	// Path is the URL path to use for the current proxy request
	Path string `json:"path,omitempty" description:"URL path to use in proxy request to pod"`
}

// Status is a return value for calls that don't return other objects.
type Status struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// One of: "Success" or "Failure"
	Status string `json:"status,omitempty" description:"status of the operation; either Success, or Failure; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
	// A human-readable description of the status of this operation.
	Message string `json:"message,omitempty" description:"human-readable description of the status of this operation"`
	// A machine-readable description of why this operation is in the
	// "Failure" status. If this value is empty there
	// is no information available. A Reason clarifies an HTTP status
	// code but does not override it.
	Reason StatusReason `json:"reason,omitempty" description:"machine-readable description of why this operation is in the 'Failure' status; if this value is empty there is no information available; a reason clarifies an HTTP status code but does not override it"`
	// Extended data associated with the reason.  Each reason may define its
	// own extended details. This field is optional and the data returned
	// is not guaranteed to conform to any schema except that defined by
	// the reason type.
	Details *StatusDetails `json:"details,omitempty" description:"extended data associated with the reason; each reason may define its own extended details; this field is optional and the data returned is not guaranteed to conform to any schema except that defined by the reason type"`
	// Suggested HTTP return code for this status, 0 if not set.
	Code int `json:"code,omitempty" description:"suggested HTTP return code for this status; 0 if not set"`
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
	Name string `json:"name,omitempty" description:"the name attribute of the resource associated with the status StatusReason (when there is a single name which can be described)"`
	// The kind attribute of the resource associated with the status StatusReason.
	// On some operations may differ from the requested resource Kind.
	Kind string `json:"kind,omitempty" description:"the kind attribute of the resource associated with the status StatusReason; on some operations may differ from the requested resource Kind; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#types-kinds"`
	// The Causes array includes more details associated with the StatusReason
	// failure. Not all StatusReasons may provide detailed causes.
	Causes []StatusCause `json:"causes,omitempty" description:"the Causes array includes more details associated with the StatusReason failure; not all StatusReasons may provide detailed causes"`
	// If specified, the time in seconds before the operation should be retried.
	RetryAfterSeconds int `json:"retryAfterSeconds,omitempty" description:"the number of seconds before the client should attempt to retry this operation"`
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
	// Status code 500
	StatusReasonServerTimeout StatusReason = "ServerTimeout"
)

// StatusCause provides more information about an api.Status failure, including
// cases when multiple errors are encountered.
type StatusCause struct {
	// A machine-readable description of the cause of the error. If this value is
	// empty there is no information available.
	Type CauseType `json:"reason,omitempty" description:"machine-readable description of the cause of the error; if this value is empty there is no information available"`
	// A human-readable description of the cause of the error.  This field may be
	// presented as-is to a reader.
	Message string `json:"message,omitempty" description:"human-readable description of the cause of the error; this field may be presented as-is to a reader"`
	// The field of the resource that has caused this error, as named by its JSON
	// serialization. May include dot and postfix notation for nested attributes.
	// Arrays are zero-indexed.  Fields may appear more than once in an array of
	// causes due to fields having multiple errors.
	// Optional.
	//
	// Examples:
	//   "name" - the field "name" on the current resource
	//   "items[0].name" - the field "name" on the first array entry in "items"
	Field string `json:"field,omitempty" description:"field of the resource that has caused this error, as named by its JSON serialization; may include dot and postfix notation for nested attributes; arrays are zero-indexed; fields may appear more than once in an array of causes due to fields having multiple errors"`
}

// CauseType is a machine readable value providing more detail about what
// occured in a status response. An operation may have multiple causes for a
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
)

// ObjectReference contains enough information to let you inspect or modify the referred object.
type ObjectReference struct {
	Kind            string    `json:"kind,omitempty" description:"kind of the referent; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#types-kinds"`
	Namespace       string    `json:"namespace,omitempty" description:"namespace of the referent; see http://releases.k8s.io/v1.0.0/docs/namespaces.md"`
	Name            string    `json:"name,omitempty" description:"name of the referent; see http://releases.k8s.io/v1.0.0/docs/identifiers.md#names"`
	UID             types.UID `json:"uid,omitempty" description:"uid of the referent; see http://releases.k8s.io/v1.0.0/docs/identifiers.md#uids"`
	APIVersion      string    `json:"apiVersion,omitempty" description:"API version of the referent"`
	ResourceVersion string    `json:"resourceVersion,omitempty" description:"specific resourceVersion to which this reference is made, if any: http://releases.k8s.io/v1.0.0/docs/api-conventions.md#concurrency-control-and-consistency"`

	// Optional. If referring to a piece of an object instead of an entire object, this string
	// should contain information to identify the sub-object. For example, if the object
	// reference is to a container within a pod, this would take on a value like:
	// "spec.containers{name}" (where "name" refers to the name of the container that triggered
	// the event) or if no container name is specified "spec.containers[2]" (container with
	// index 2 in this pod). This syntax is chosen only to have some well-defined way of
	// referencing a part of an object.
	// TODO: this design is not final and this field is subject to change in the future.
	FieldPath string `json:"fieldPath,omitempty" description:"if referring to a piece of an object instead of an entire object, this string should contain a valid JSON/Go field access statement, such as desiredState.manifest.containers[2]"`
}

// LocalObjectReference contains enough information to let you locate the referenced object inside the same namespace.
type LocalObjectReference struct {
	//TODO: Add other useful fields.  apiVersion, kind, uid?
	Name string `json:"name,omitempty" description:"name of the referent; see http://releases.k8s.io/v1.0.0/docs/identifiers.md#names"`
}

type SerializedReference struct {
	TypeMeta  `json:",inline"`
	Reference ObjectReference `json:"reference,omitempty" description:"the reference to an object in the system"`
}

type EventSource struct {
	// Component from which the event is generated.
	Component string `json:"component,omitempty" description:"component that generated the event"`
	// Host name on which the event is generated.
	Host string `json:"host,omitempty" description:"name of the host where the event is generated"`
}

// Event is a report of an event somewhere in the cluster.
// TODO: Decide whether to store these separately or with the object they apply to.
type Event struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Required. The object that this event is about.
	InvolvedObject ObjectReference `json:"involvedObject" description:"object this event is about"`

	// Optional; this should be a short, machine understandable string that gives the reason
	// for this event being generated.
	// TODO: provide exact specification for format.
	Reason string `json:"reason,omitempty" description:"short, machine understandable string that gives the reason for the transition into the object's current status"`

	// Optional. A human-readable description of the status of this operation.
	// TODO: decide on maximum length.
	Message string `json:"message,omitempty" description:"human-readable description of the status of this operation"`

	// Optional. The component reporting this event. Should be a short machine understandable string.
	Source EventSource `json:"source,omitempty" description:"component reporting this event"`

	// The time at which the event was first recorded. (Time of server receipt is in TypeMeta.)
	FirstTimestamp util.Time `json:"firstTimestamp,omitempty" description:"the time at which the event was first recorded"`

	// The time at which the most recent occurance of this event was recorded.
	LastTimestamp util.Time `json:"lastTimestamp,omitempty" description:"the time at which the most recent occurance of this event was recorded"`

	// The number of times this event has occurred.
	Count int `json:"count,omitempty" description:"the number of times this event has occurred"`
}

// EventList is a list of events.
type EventList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []Event `json:"items" description:"list of events"`
}

// List holds a list of objects, which may not be known by the server.
type List struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []runtime.RawExtension `json:"items" description:"list of objects"`
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
	Type LimitType `json:"type,omitempty" description:"type of resource that this limit applies to"`
	// Max usage constraints on this kind by resource name
	Max ResourceList `json:"max,omitempty" description:"max usage constraints on this kind by resource name"`
	// Min usage constraints on this kind by resource name
	Min ResourceList `json:"min,omitempty" description:"min usage constraints on this kind by resource name"`
	// Default usage constraints on this kind by resource name
	Default ResourceList `json:"default,omitempty" description:"default values on this kind by resource name if omitted"`
}

// LimitRangeSpec defines a min/max usage limit for resources that match on kind
type LimitRangeSpec struct {
	// Limits is the list of LimitRangeItem objects that are enforced
	Limits []LimitRangeItem `json:"limits" description:"limits is the list of LimitRangeItem objects that are enforced"`
}

// LimitRange sets resource usage limits for each kind of resource in a Namespace
type LimitRange struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the limits enforced
	Spec LimitRangeSpec `json:"spec,omitempty" description:"spec defines the limits enforced; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// LimitRangeList is a list of LimitRange items.
type LimitRangeList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Items is a list of LimitRange objects
	Items []LimitRange `json:"items" description:"items is a list of LimitRange objects; see http://releases.k8s.io/v1.0.0/docs/design/admission_control_limit_range.md"`
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
	// ResourcePersistentVolumeClaims, number
	ResourcePersistentVolumeClaims ResourceName = "persistentvolumeclaims"
)

// ResourceQuotaSpec defines the desired hard limits to enforce for Quota
type ResourceQuotaSpec struct {
	// Hard is the set of desired hard limits for each named resource
	Hard ResourceList `json:"hard,omitempty" description:"hard is the set of desired hard limits for each named resource; see http://releases.k8s.io/v1.0.0/docs/design/admission_control_resource_quota.md#admissioncontrol-plugin-resourcequota"`
}

// ResourceQuotaStatus defines the enforced hard limits and observed use
type ResourceQuotaStatus struct {
	// Hard is the set of enforced hard limits for each named resource
	Hard ResourceList `json:"hard,omitempty" description:"hard is the set of enforced hard limits for each named resource; see http://releases.k8s.io/v1.0.0/docs/design/admission_control_resource_quota.md#admissioncontrol-plugin-resourcequota"`
	// Used is the current observed total usage of the resource in the namespace
	Used ResourceList `json:"used,omitempty" description:"used is the current observed total usage of the resource in the namespace"`
}

// ResourceQuota sets aggregate quota restrictions enforced per namespace
type ResourceQuota struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Spec defines the desired quota
	Spec ResourceQuotaSpec `json:"spec,omitempty" description:"spec defines the desired quota; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`

	// Status defines the actual enforced quota and its current usage
	Status ResourceQuotaStatus `json:"status,omitempty" description:"status defines the actual enforced quota and current usage; http://releases.k8s.io/v1.0.0/docs/api-conventions.md#spec-and-status"`
}

// ResourceQuotaList is a list of ResourceQuota items
type ResourceQuotaList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Items is a list of ResourceQuota objects
	Items []ResourceQuota `json:"items" description:"items is a list of ResourceQuota objects; see http://releases.k8s.io/v1.0.0/docs/design/admission_control_resource_quota.md#admissioncontrol-plugin-resourcequota"`
}

// Secret holds secret data of a certain type.  The total bytes of the values in
// the Data field must be less than MaxSecretSize bytes.
type Secret struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	// Data contains the secret data.  Each key must be a valid DNS_SUBDOMAIN
	// or leading dot followed by valid DNS_SUBDOMAIN.
	// The serialized form of the secret data is a base64 encoded string,
	// representing the arbitrary (possibly non-string) data value here.
	Data map[string][]byte `json:"data,omitempty" description:"data contains the secret data.  Each key must be a valid DNS_SUBDOMAIN or leading dot followed by valid DNS_SUBDOMAIN.  Each value must be a base64 encoded string as described in https://tools.ietf.org/html/rfc4648#section-4"`

	// Used to facilitate programmatic handling of secret data.
	Type SecretType `json:"type,omitempty" description:"type facilitates programmatic handling of secret data"`
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
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []Secret `json:"items" description:"items is a list of secret objects; see http://releases.k8s.io/v1.0.0/docs/secrets.md"`
}

// Type and constants for component health validation.
type ComponentConditionType string

// These are the valid conditions for the component.
const (
	ComponentHealthy ComponentConditionType = "Healthy"
)

type ComponentCondition struct {
	Type    ComponentConditionType `json:"type" description:"type of component condition, currently only Healthy"`
	Status  ConditionStatus        `json:"status" description:"current status of this component condition, one of True, False, Unknown"`
	Message string                 `json:"message,omitempty" description:"health check message received from the component"`
	Error   string                 `json:"error,omitempty" description:"error code from health check attempt (if any)"`
}

// ComponentStatus (and ComponentStatusList) holds the cluster validation info.
type ComponentStatus struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Conditions []ComponentCondition `json:"conditions,omitempty" description:"list of component conditions observed" patchStrategy:"merge" patchMergeKey:"type"`
}

type ComponentStatusList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Items []ComponentStatus `json:"items" description:"list of component status objects"`
}

// SecurityContext holds security configuration that will be applied to a container.  SecurityContext
// contains duplication of some existing fields from the Container resource.  These duplicate fields
// will be populated based on the Container configuration if they are not set.  Defining them on
// both the Container AND the SecurityContext will result in an error.
type SecurityContext struct {
	// Capabilities are the capabilities to add/drop when running the container
	// Must match Container.Capabilities or be unset.  Will be defaulted to Container.Capabilities if left unset
	Capabilities *Capabilities `json:"capabilities,omitempty" description:"the linux capabilites that should be added or removed; see http://releases.k8s.io/v1.0.0/docs/design/security_context.md#security-context"`

	// Run the container in privileged mode
	// Must match Container.Privileged or be unset.  Will be defaulted to Container.Privileged if left unset
	Privileged *bool `json:"privileged,omitempty" description:"run the container in privileged mode; see http://releases.k8s.io/v1.0.0/docs/design/security_context.md#security-context"`

	// SELinuxOptions are the labels to be applied to the container
	// and volumes
	SELinuxOptions *SELinuxOptions `json:"seLinuxOptions,omitempty" description:"options that control the SELinux labels applied; see http://releases.k8s.io/v1.0.0/docs/design/security_context.md#security-context"`

	// RunAsUser is the UID to run the entrypoint of the container process.
	RunAsUser *int64 `json:"runAsUser,omitempty" description:"the user id that runs the first process in the container; see http://releases.k8s.io/v1.0.0/docs/design/security_context.md#security-context"`
}

// SELinuxOptions are the labels to be applied to the container
type SELinuxOptions struct {
	// SELinux user label
	User string `json:"user,omitempty" description:"the user label to apply to the container; see http://releases.k8s.io/v1.0.0/docs/labels.md"`

	// SELinux role label
	Role string `json:"role,omitempty" description:"the role label to apply to the container; see http://releases.k8s.io/v1.0.0/docs/labels.md"`

	// SELinux type label
	Type string `json:"type,omitempty" description:"the type label to apply to the container; see http://releases.k8s.io/v1.0.0/docs/labels.md"`

	// SELinux level label.
	Level string `json:"level,omitempty" description:"the level label to apply to the container; see http://releases.k8s.io/v1.0.0/docs/labels.md"`
}

// RangeAllocation is not a public type
type RangeAllocation struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/v1.0.0/docs/api-conventions.md#metadata"`

	Range string `json:"range" description:"a range string that identifies the range represented by 'data'; required"`
	Data  []byte `json:"data" description:"a bit array containing all allocated addresses in the previous segment"`
}
