/*
Copyright 2014 Google Inc. All rights reserved.

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

	// CreationTimestamp is a timestamp representing the server time when this object was
	// created. It is not guaranteed to be set in happens-before order across separate operations.
	// Clients may not set this value. It is represented in RFC3339 form and is in UTC.
	CreationTimestamp util.Time `json:"creationTimestamp,omitempty"`

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
	// TerminationMessagePathDefault means the default path to capture the application termination message running in a container
	TerminationMessagePathDefault string = "/dev/termination-log"
)

// Volume represents a named volume in a pod that may be accessed by any containers in the pod.
type Volume struct {
	// Required: This must be a DNS_LABEL.  Each volume in a pod must have
	// a unique name.
	Name string `json:"name"`
	// Source represents the location and type of a volume to mount.
	// This is optional for now. If not specified, the Volume is implied to be an EmptyDir.
	// This implied behavior is deprecated and will be removed in a future version.
	Source VolumeSource `json:"source,omitempty"`
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
	HostPath *HostPathVolumeSource `json:"hostPath"`
	// EmptyDir represents a temporary directory that shares a pod's lifetime.
	EmptyDir *EmptyDirVolumeSource `json:"emptyDir"`
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDiskVolumeSource `json:"persistentDisk"`
	// GitRepo represents a git repository at a particular revision.
	GitRepo *GitRepoVolumeSource `json:"gitRepo"`
	// Secret represents a secret that should populate this volume.
	Secret *SecretVolumeSource `json:"secret"`
}

// HostPathVolumeSource represents bare host directory volume.
type HostPathVolumeSource struct {
	Path string `json:"path"`
}

type EmptyDirVolumeSource struct{}

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
	// Reference to a Secret
	Target ObjectReference `json:"target"`

	// EnvAdaptations adapts the secret into an environment file
	EnvAdaptations *SecretEnv `json:"env"`
	// FUTURE: fine-grained control of files
	// FileAdaptations *SecretFiles `json:"files"`
}

// SecretEnv adapts secret data into an environment file.  The keys in the
// secret data are converted from DNS_SUBDOMAIN into environment variable
// names by upper casing and subsituting the '_' character for '-' and '.'.
type SecretEnv struct {
	// Name is the name of the environment file to generate.
	Name string `json:"name"`

	// Adaptations is a list of adaptations to apply to the
	// names of environment variables superceding the normal
	// algorithm.
	Adaptations []StringAdaptation `json:"adaptations,omitempty"`
}

// StringAdaptation specifies a mapping from one value to another
type StringAdaptation struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// Port represents a network port in a single container
type Port struct {
	// Optional: If specified, this must be a DNS_LABEL.  Each named port
	// in a pod must have a unique name.
	Name string `json:"name,omitempty"`
	// Optional: If specified, this must be a valid port number, 0 < x < 65536.
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
	// Optional: defaults to "".
	Value string `json:"value,omitempty"`
}

// HTTPGetAction describes an action based on HTTP Get requests.
type HTTPGetAction struct {
	// Optional: Path to access on the HTTP server.
	Path string `json:"path,omitempty"`
	// Required: Name or number of the port to access on the container.
	Port util.IntOrString `json:"port,omitempty"`
	// Optional: Host name to connect to, defaults to the pod IP.
	Host string `json:"host,omitempty"`
}

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

// CapabilityType represent POSIX capabilities type
type CapabilityType string

// Capabilities represent POSIX capabilities that can be added or removed to a running container.
type Capabilities struct {
	// Added capabilities
	Add []CapabilityType `json:"add,omitempty"`
	// Removed capabilities
	Drop []CapabilityType `json:"drop,omitempty"`
}

// ResourceRequirements describes the compute resource requirements.
type ResourceRequirements struct {
	// Limits describes the maximum amount of compute resources required.
	Limits ResourceList `json:"limits,omitempty"`
}

// Container represents a single container that is expected to be run on the host.
type Container struct {
	// Required: This must be a DNS_LABEL.  Each container in a pod must
	// have a unique name.
	Name string `json:"name"`
	// Required.
	Image string `json:"image"`
	// Optional: Defaults to whatever is defined in the image.
	Command []string `json:"command,omitempty"`
	// Optional: Defaults to Docker's default.
	WorkingDir string   `json:"workingDir,omitempty"`
	Ports      []Port   `json:"ports,omitempty"`
	Env        []EnvVar `json:"env,omitempty"`
	// Compute resource requirements.
	Resources      ResourceRequirements `json:"resources,omitempty"`
	VolumeMounts   []VolumeMount        `json:"volumeMounts,omitempty"`
	LivenessProbe  *Probe               `json:"livenessProbe,omitempty"`
	ReadinessProbe *Probe               `json:"readinessProbe,omitempty"`
	Lifecycle      *Lifecycle           `json:"lifecycle,omitempty"`
	// Required.
	TerminationMessagePath string `json:"terminationMessagePath,omitempty"`
	// Optional: Default to false.
	Privileged bool `json:"privileged,omitempty"`
	// Required: Policy for pulling images for this container
	ImagePullPolicy PullPolicy `json:"imagePullPolicy"`
	// Optional: Capabilities for container.
	Capabilities Capabilities `json:"capabilities,omitempty"`
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

// These are valid condition statuses. "ConditionFull" means a resource is in the condition;
// "ConditionNone" means a resource is not in the condition; "ConditionUnknown" means kubernetes
// can't decide if a resource is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
const (
	ConditionFull    ConditionStatus = "Full"
	ConditionNone    ConditionStatus = "None"
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
	ExitCode   int       `json:"exitCode"`
	Signal     int       `json:"signal,omitempty"`
	Reason     string    `json:"reason,omitempty"`
	Message    string    `json:"message,omitempty"`
	StartedAt  util.Time `json:"startedAt,omitempty"`
	FinishedAt util.Time `json:"finishedAt,omitempty"`
}

// ContainerState holds a possible state of container.
// Only one of its members may be specified.
// If none of them is specified, the default one is ContainerStateWaiting.
type ContainerState struct {
	Waiting     *ContainerStateWaiting    `json:"waiting,omitempty"`
	Running     *ContainerStateRunning    `json:"running,omitempty"`
	Termination *ContainerStateTerminated `json:"termination,omitempty"`
}

type ContainerStatus struct {
	// TODO(dchen1107): Should we rename PodStatus to a more generic name or have a separate states
	// defined for container?
	State ContainerState `json:"state,omitempty"`
	// Ready specifies whether the conatiner has passed its readiness check.
	Ready bool `json:"ready"`
	// Note that this is calculated from dead containers.  But those containers are subject to
	// garbage collection.  This value will get capped at 5 by GC.
	RestartCount int `json:"restartCount"`
	// TODO(dchen1107): Deprecated this soon once we pull entire PodStatus from node,
	// not just PodInfo. Now we need this to remove docker.Container from API
	PodIP string `json:"podIP,omitempty"`
	// TODO(dchen1107): Need to decide how to represent this in v1beta3
	Image       string `json:"image"`
	ImageID     string `json:"imageID" description:"ID of the container's image"`
	ContainerID string `json:"containerID,omitempty" description:"container's ID in the format 'docker://<container_id>'"`
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

// PodInfo contains one entry for every container with available info.
type PodInfo map[string]ContainerStatus

// PodContainerInfo is a wrapper for PodInfo that can be encode/decoded
// DEPRECATED: Replaced with PodStatusResult
type PodContainerInfo struct {
	TypeMeta      `json:",inline"`
	ObjectMeta    `json:"metadata,omitempty"`
	ContainerInfo PodInfo `json:"containerInfo"`
}

type RestartPolicyAlways struct{}

// TODO(dchen1107): Define what kinds of failures should restart.
// TODO(dchen1107): Decide whether to support policy knobs, and, if so, which ones.
type RestartPolicyOnFailure struct{}

type RestartPolicyNever struct{}

// RestartPolicy describes how the container should be restarted.
// Only one of the following restart policies may be specified.
// If none of the following policies is specified, the default one
// is RestartPolicyAlways.
type RestartPolicy struct {
	Always    *RestartPolicyAlways    `json:"always,omitempty"`
	OnFailure *RestartPolicyOnFailure `json:"onFailure,omitempty"`
	Never     *RestartPolicyNever     `json:"never,omitempty"`
}

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
	Volumes       []Volume      `json:"volumes"`
	Containers    []Container   `json:"containers"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty"`
	// Required: Set DNS policy.
	DNSPolicy DNSPolicy `json:"dnsPolicy,omitempty"`
	// NodeSelector is a selector which must be true for the pod to fit on a node
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Host is a request to schedule this pod onto a specific host.  If it is non-empty,
	// the the scheduler simply schedules this pod onto that host, assuming that it fits
	// resource requirements.
	Host string `json:"host,omitempty"`
}

// PodStatus represents information about the status of a pod. Status may trail the actual
// state of a system.
type PodStatus struct {
	Phase      PodPhase       `json:"phase,omitempty"`
	Conditions []PodCondition `json:"Condition,omitempty"`
	// A human readable message indicating details about why the pod is in this state.
	Message string `json:"message,omitempty"`

	// Host is the name of the node that this Pod is currently bound to, or empty if no
	// assignment has been done.
	Host   string `json:"host,omitempty"`
	HostIP string `json:"hostIP,omitempty"`
	PodIP  string `json:"podIP,omitempty"`

	// The key of this map is the *name* of the container within the manifest; it has one
	// entry per container in the manifest. The value of this map is currently the output
	// of `docker inspect`. This output format is *not* final and should not be relied
	// upon.
	// TODO: Make real decisions about what our info should look like. Re-enable fuzz test
	// when we have done this.
	Info PodInfo `json:"info,omitempty"`
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

	// Spec defines the pods that will be created from this template
	Spec PodTemplateSpec `json:"spec,omitempty"`
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
	// Must be set before converting to a v1beta3 API object
	TemplateRef *ObjectReference `json:"templateRef,omitempty"`

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

// ServiceList holds a list of services.
type ServiceList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Service `json:"items"`
}

// Session Affinity Type string
type AffinityType string

const (
	// AffinityTypeClientIP is the Client IP based.
	AffinityTypeClientIP AffinityType = "ClientIP"

	// AffinityTypeNone - no session affinity.
	AffinityTypeNone AffinityType = "None"
)

// ServiceStatus represents the current status of a service
type ServiceStatus struct{}

// ServiceSpec describes the attributes that a user creates on a service
type ServiceSpec struct {
	// Port is the TCP or UDP port that will be made available to each pod for connecting to the pods
	// proxied by this service.
	Port int `json:"port"`

	// Required: Supports "TCP" and "UDP".
	Protocol Protocol `json:"protocol,omitempty"`

	// This service will route traffic to pods having labels matching this selector. If empty or not present,
	// the service is assumed to have endpoints set by an external process and Kubernetes will not modify
	// those endpoints.
	Selector map[string]string `json:"selector"`

	// PortalIP is usually assigned by the master.  If specified by the user
	// we will try to respect it or else fail the request.  This field can
	// not be changed by updates.
	PortalIP string `json:"portalIP,omitempty"`

	// CreateExternalLoadBalancer indicates whether a load balancer should be created for this service.
	CreateExternalLoadBalancer bool `json:"createExternalLoadBalancer,omitempty"`
	// PublicIPs are used by external load balancers.
	PublicIPs []string `json:"publicIPs,omitempty"`

	// ContainerPort is the name or number of the port on the container to direct traffic to.
	// This is useful if the containers the service points to have multiple open ports.
	// Optional: If unspecified, the first port on the container will be used.
	ContainerPort util.IntOrString `json:"containerPort,omitempty"`

	// Required: Supports "ClientIP" and "None".  Used to maintain session affinity.
	SessionAffinity AffinityType `json:"sessionAffinity,omitempty"`
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

// Endpoints is a collection of endpoints that implement the actual service, for example:
// Name: "mysql", Endpoints: [{"ip": "10.10.1.1", "port": 1909}, {"ip": "10.10.2.2", "port": 8834}]
type Endpoints struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Optional: The IP protocol for these endpoints. Supports "TCP" and
	// "UDP".  Defaults to "TCP".
	Protocol  Protocol   `json:"protocol,omitempty"`
	Endpoints []Endpoint `json:"endpoints,omitempty"`
}

// Endpoint is a single IP endpoint of a service.
type Endpoint struct {
	// Required: The IP of this endpoint.
	// TODO: This should allow hostname or IP, see #4447.
	IP string `json:"ip"`

	// Required: The destination port to access.
	Port int `json:"port"`
}

// EndpointsList is a list of endpoints.
type EndpointsList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Endpoints `json:"items"`
}

// NodeSpec describes the attributes that a node is created with.
type NodeSpec struct {
	// Capacity represents the available resources of a node
	Capacity ResourceList `json:"capacity,omitempty"`
	// PodCIDR represents the pod IP range assigned to the node
	// Note: assigning IP ranges to nodes might need to be revisited when we support migratable IPs.
	PodCIDR string `json:"cidr,omitempty"`
}

// NodeStatus is information about the current status of a node.
type NodeStatus struct {
	// Queried from cloud provider, if available.
	HostIP string `json:"hostIP,omitempty"`
	// NodePhase is the current lifecycle phase of the node.
	Phase NodePhase `json:"phase,omitempty"`
	// Conditions is an array of current node conditions.
	Conditions []NodeCondition `json:"conditions,omitempty"`
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
	// NodeReachable means the node can be reached (in the sense of HTTP connection) from node controller.
	NodeReachable NodeConditionType = "Reachable"
	// NodeReady means the node returns StatusOK for HTTP health check.
	NodeReady NodeConditionType = "Ready"
)

type NodeCondition struct {
	Type               NodeConditionType `json:"type"`
	Status             ConditionStatus   `json:"status"`
	LastProbeTime      util.Time         `json:"lastProbeTime,omitempty"`
	LastTransitionTime util.Time         `json:"lastTransitionTime,omitempty"`
	Reason             string            `json:"reason,omitempty"`
	Message            string            `json:"message,omitempty"`
}

// NodeResources is an object for conveying resource information about a node.
// see https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/resources.md for more details.
// TODO: Use ResourceList instead?
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

// NodeList is a list of minions.
type NodeList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []Node `json:"items"`
}

// NamespaceSpec describes the attributes on a Namespace
type NamespaceSpec struct {
}

// NamespaceStatus is information about the current status of a Namespace.
type NamespaceStatus struct {
}

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

// Binding is written by a scheduler to cause a pod to be bound to a host.
type Binding struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	PodID string `json:"podID"`
	Host  string `json:"host"`
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
	// The ID attribute of the resource associated with the status StatusReason
	// (when there is a single ID which can be described).
	// TODO: replace with Name with v1beta3
	ID string `json:"id,omitempty"`
	// The kind attribute of the resource associated with the status StatusReason.
	// On some operations may differ from the requested resource Kind.
	Kind string `json:"kind,omitempty"`
	// The Causes array includes more details associated with the StatusReason
	// failure. Not all StatusReasons may provide detailed causes.
	Causes []StatusCause `json:"causes,omitempty"`
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
	// Status code 500
	StatusReasonServerTimeout StatusReason = "ServerTimeout"

	// StatusReasonTimeout means that the request could not be completed within the given time.
	// Clients can get this response only when they specified a timeout param in the request.
	// The request might succeed with an increased value of timeout param.
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

	// The time at which the most recent occurance of this event was recorded.
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

// ContainerManifest corresponds to the Container Manifest format, documented at:
// https://developers.google.com/compute/docs/containers/container_vms#container_manifest
// This is used as the representation of Kubernetes workloads.
// DEPRECATED: Replaced with BoundPod
type ContainerManifest struct {
	// Required: This must be a supported version string, such as "v1beta1".
	Version string `json:"version"`
	// Required: This must be a DNS_SUBDOMAIN.
	// TODO: ID on Manifest is deprecated and will be removed in the future.
	ID string `json:"id"`
	// TODO: UUID on Manifest is deprecated in the future once we are done
	// with the API refactoring. It is required for now to determine the instance
	// of a Pod.
	UUID          types.UID     `json:"uuid,omitempty"`
	Volumes       []Volume      `json:"volumes"`
	Containers    []Container   `json:"containers"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty"`
	// Required: Set DNS policy.
	DNSPolicy DNSPolicy `json:"dnsPolicy"`
}

// ContainerManifestList is used to communicate container manifests to kubelet.
// DEPRECATED: Replaced with BoundPods
type ContainerManifestList struct {
	TypeMeta `json:",inline"`
	ListMeta `json:"metadata,omitempty"`

	Items []ContainerManifest `json:"items"`
}

// BoundPod is a collection of containers that should be run on a host. A BoundPod
// defines how a Pod may change after a Binding is created. A Pod is a request to
// execute a pod, whereas a BoundPod is the specification that would be run on a server.
type BoundPod struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behavior of a pod.
	Spec PodSpec `json:"spec,omitempty"`
}

// BoundPods is a list of Pods bound to a common server. The resource version of
// the pod list is guaranteed to only change when the list of bound pods changes.
type BoundPods struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

	// Host is the name of a node that these pods were bound to.
	Host string `json:"host"`

	// Items is the list of all pods bound to a given host.
	Items []BoundPod `json:"items"`
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
	// ResourceQuotas, number
	ResourceQuotas ResourceName = "resourcequotas"
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

// ResourceQuotaUsage captures system observed quota status per namespace
// It is used to enforce atomic updates of a backing ResourceQuota.Status field in storage
type ResourceQuotaUsage struct {
	TypeMeta   `json:",inline"`
	ObjectMeta `json:"metadata,omitempty"`

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

	// Data contains the secret data.  Each key must be a valid DNS_SUBDOMAIN.
	// The serialized form of the secret data is a base64 encoded string,
	// representing the arbitrary (possibly non-string) data value here.
	Data map[string][]byte `json:"data,omitempty"`

	// Used to facilitate programatic handling of secret data.
	Type SecretType `json:"type,omitempty"`
}

const MaxSecretSize = 1 * 1024 * 1024

type SecretType string

const (
	SecretTypeOpaque SecretType = "Opaque" // Default; arbitrary user-defined data
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
