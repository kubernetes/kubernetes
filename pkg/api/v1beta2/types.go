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

package v1beta2

import (
	"time"

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

// Volume represents a named volume in a pod that may be accessed by any containers in the pod.
type Volume struct {
	// Required: This must be a DNS_LABEL.  Each volume in a pod must have
	// a unique name.
	Name string `yaml:"name" json:"name"`
	// Source represents the location and type of a volume to mount.
	// This is optional for now. If not specified, the Volume is implied to be an EmptyDir.
	// This implied behavior is deprecated and will be removed in a future version.
	Source *VolumeSource `yaml:"source" json:"source"`
}

type VolumeSource struct {
	// Only one of the following sources may be specified
	// HostDir represents a pre-existing directory on the host machine that is directly
	// exposed to the container. This is generally used for system agents or other privileged
	// things that are allowed to see the host machine. Most containers will NOT need this.
	// TODO(jonesdl) We need to restrict who can use host directory mounts and
	// who can/can not mount host directories as read/write.
	HostDir *HostDir `yaml:"hostDir" json:"hostDir"`
	// EmptyDir represents a temporary directory that shares a pod's lifetime.
	EmptyDir *EmptyDir `yaml:"emptyDir" json:"emptyDir"`
	// A persistent disk that is mounted to the
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDisk `yaml:"persistentDisk" json:"persistentDisk"`
	// GitRepo represents a git repository at a particular revision.
	GitRepo *GitRepo `json:"gitRepo" yaml:"gitRepo"`
}

// HostDir represents bare host directory volume.
type HostDir struct {
	Path string `yaml:"path" json:"path"`
}

type EmptyDir struct{}

// Protocol defines network protocols supported for things like conatiner ports.
type Protocol string

const (
	// ProtocolTCP is the TCP protocol.
	ProtocolTCP Protocol = "TCP"
	// ProtocolUDP is the UDP protocol.
	ProtocolUDP Protocol = "UDP"
)

// Port represents a network port in a single container.
type Port struct {
	// Optional: If specified, this must be a DNS_LABEL.  Each named port
	// in a pod must have a unique name.
	Name string `yaml:"name,omitempty" json:"name,omitempty"`
	// Optional: If specified, this must be a valid port number, 0 < x < 65536.
	HostPort int `yaml:"hostPort,omitempty" json:"hostPort,omitempty"`
	// Required: This must be a valid port number, 0 < x < 65536.
	ContainerPort int `yaml:"containerPort" json:"containerPort"`
	// Optional: Defaults to "TCP".
	Protocol Protocol `yaml:"protocol,omitempty" json:"protocol,omitempty"`
	// Optional: What host IP to bind the external port to.
	HostIP string `yaml:"hostIP,omitempty" json:"hostIP,omitempty"`
}

// GCEPersistent Disk resource.
// A GCE PD must exist before mounting to a container. The disk must
// also be in the same GCE project and zone as the kubelet.
// A GCE PD can only be mounted as read/write once.
type GCEPersistentDisk struct {
	// Unique name of the PD resource. Used to identify the disk in GCE
	PDName string `yaml:"pdName" json:"pdName"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	FSType string `yaml:"fsType,omitempty" json:"fsType,omitempty"`
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field 0 or empty.
	Partition int `yaml:"partition,omitempty" json:"partition,omitempty"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `yaml:"readOnly,omitempty" json:"readOnly,omitempty"`
}

// GitRepo represents a volume that is pulled from git when the pod is created.
type GitRepo struct {
	// Repository URL
	Repository string `yaml:"repository" json:"repository"`
	// Commit hash, this is optional
	Revision string `yaml:"revision" json:"revision"`
}

// VolumeMount describes a mounting of a Volume within a container.
type VolumeMount struct {
	// Required: This must match the Name of a Volume [above].
	Name string `yaml:"name" json:"name"`
	// Optional: Defaults to false (read-write).
	ReadOnly bool `yaml:"readOnly,omitempty" json:"readOnly,omitempty"`
	// Required.
	MountPath string `yaml:"mountPath,omitempty" json:"mountPath,omitempty"`
}

// EnvVar represents an environment variable present in a Container.
type EnvVar struct {
	// Required: This must be a C_IDENTIFIER.
	Name string `yaml:"name" json:"name"`
	// Optional: defaults to "".
	Value string `yaml:"value,omitempty" json:"value,omitempty"`
}

// HTTPGetAction describes an action based on HTTP Get requests.
type HTTPGetAction struct {
	// Optional: Path to access on the HTTP server.
	Path string `yaml:"path,omitempty" json:"path,omitempty"`
	// Required: Name or number of the port to access on the container.
	Port util.IntOrString `yaml:"port,omitempty" json:"port,omitempty"`
	// Optional: Host name to connect to, defaults to the pod IP.
	Host string `yaml:"host,omitempty" json:"host,omitempty"`
}

// TCPSocketAction describes an action based on opening a socket
type TCPSocketAction struct {
	// Required: Port to connect to.
	Port util.IntOrString `yaml:"port,omitempty" json:"port,omitempty"`
}

// ExecAction describes a "run in container" action.
type ExecAction struct {
	// Command is the command line to execute inside the container, the working directory for the
	// command  is root ('/') in the container's filesystem.  The command is simply exec'd, it is
	// not run inside a shell, so traditional shell instructions ('|', etc) won't work.  To use
	// a shell, you need to explicitly call out to that shell.
	Command []string `yaml:"command,omitempty" json:"command,omitempty"`
}

// LivenessProbe describes a liveness probe to be examined to the container.
// TODO: pass structured data to the actions, and document that data here.
type LivenessProbe struct {
	// HTTPGetProbe parameters, required if Type == 'http'
	HTTPGet *HTTPGetAction `yaml:"httpGet,omitempty" json:"httpGet,omitempty"`
	// TCPSocketProbe parameter, required if Type == 'tcp'
	TCPSocket *TCPSocketAction `yaml:"tcpSocket,omitempty" json:"tcpSocket,omitempty"`
	// ExecProbe parameter, required if Type == 'exec'
	Exec *ExecAction `yaml:"exec,omitempty" json:"exec,omitempty"`
	// Length of time before health checking is activated.  In seconds.
	InitialDelaySeconds int64 `yaml:"initialDelaySeconds,omitempty" json:"initialDelaySeconds,omitempty"`
}

// PullPolicy describes a policy for if/when to pull a container image
type PullPolicy string

const (
	// Always attempt to pull the latest image.  Container will fail If the pull fails.
	PullAlways PullPolicy = "PullAlways"
	// Never pull an image, only use a local image.  Container will fail if the image isn't present
	PullNever PullPolicy = "PullNever"
	// Pull if the image isn't present on disk. Container will fail if the image isn't present and the pull fails.
	PullIfNotPresent PullPolicy = "PullIfNotPresent"
)

// Container represents a single container that is expected to be run on the host.
type Container struct {
	// Required: This must be a DNS_LABEL.  Each container in a pod must
	// have a unique name.
	Name string `yaml:"name" json:"name"`
	// Required.
	Image string `yaml:"image" json:"image"`
	// Optional: Defaults to whatever is defined in the image.
	Command []string `yaml:"command,omitempty" json:"command,omitempty"`
	// Optional: Defaults to Docker's default.
	WorkingDir string   `yaml:"workingDir,omitempty" json:"workingDir,omitempty"`
	Ports      []Port   `yaml:"ports,omitempty" json:"ports,omitempty"`
	Env        []EnvVar `yaml:"env,omitempty" json:"env,omitempty"`
	// Optional: Defaults to unlimited.
	Memory int `yaml:"memory,omitempty" json:"memory,omitempty"`
	// Optional: Defaults to unlimited.
	CPU           int            `yaml:"cpu,omitempty" json:"cpu,omitempty"`
	VolumeMounts  []VolumeMount  `yaml:"volumeMounts,omitempty" json:"volumeMounts,omitempty"`
	LivenessProbe *LivenessProbe `yaml:"livenessProbe,omitempty" json:"livenessProbe,omitempty"`
	Lifecycle     *Lifecycle     `yaml:"lifecycle,omitempty" json:"lifecycle,omitempty"`
	// Optional: Defaults to /dev/termination-log
	TerminationMessagePath string `yaml:"terminationMessagePath,omitempty" json:"terminationMessagePath,omitempty"`
	// Optional: Default to false.
	Privileged bool `json:"privileged,omitempty" yaml:"privileged,omitempty"`
	// Optional: Policy for pulling images for this container
	ImagePullPolicy PullPolicy `json:"imagePullPolicy" yaml:"imagePullPolicy"`
}

// Handler defines a specific action that should be taken
// TODO: pass structured data to these actions, and document that data here.
type Handler struct {
	// One and only one of the following should be specified.
	// Exec specifies the action to take.
	Exec *ExecAction `yaml:"exec,omitempty" json:"exec,omitempty"`
	// HTTPGet specifies the http request to perform.
	HTTPGet *HTTPGetAction `yaml:"httpGet,omitempty" json:"httpGet,omitempty"`
}

// Lifecycle describes actions that the management system should take in response to container lifecycle
// events.  For the PostStart and PreStop lifecycle handlers, management of the container blocks
// until the action is complete, unless the container process fails, in which case the handler is aborted.
type Lifecycle struct {
	// PostStart is called immediately after a container is created.  If the handler fails, the container
	// is terminated and restarted.
	PostStart *Handler `yaml:"postStart,omitempty" json:"postStart,omitempty"`
	// PreStop is called immediately before a container is terminated.  The reason for termination is
	// passed to the handler.  Regardless of the outcome of the handler, the container is eventually terminated.
	PreStop *Handler `yaml:"preStop,omitempty" json:"preStop,omitempty"`
}

// The below types are used by kube_client and api_server.

// TypeMeta is shared by all objects sent to, or returned from the client.
type TypeMeta struct {
	Kind              string    `json:"kind,omitempty" yaml:"kind,omitempty"`
	ID                string    `json:"id,omitempty" yaml:"id,omitempty"`
	UID               string    `json:"uid,omitempty" yaml:"uid,omitempty"`
	CreationTimestamp util.Time `json:"creationTimestamp,omitempty" yaml:"creationTimestamp,omitempty"`
	SelfLink          string    `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
	ResourceVersion   uint64    `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
	APIVersion        string    `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	Namespace         string    `json:"namespace,omitempty" yaml:"namespace,omitempty"`

	// Annotations are unstructured key value data stored with a resource that may be set by
	// external tooling. They are not queryable and should be preserved when modifying
	// objects.
	Annotations map[string]string `json:"annotations,omitempty" yaml:"annotations,omitempty"`
}

// PodStatus represents a status of a pod.
type PodStatus string

// These are the valid statuses of pods.
const (
	// PodWaiting means that we're waiting for the pod to begin running.
	PodWaiting PodStatus = "Waiting"
	// PodRunning means that the pod is up and running.
	PodRunning PodStatus = "Running"
	// PodTerminated means that the pod has stopped.
	PodTerminated PodStatus = "Terminated"
)

type ContainerStateWaiting struct {
	// Reason could be pulling image,
	Reason string `json:"reason,omitempty" yaml:"reason,omitempty"`
}

type ContainerStateRunning struct {
	StartedAt time.Time `json:"startedAt,omitempty" yaml:"startedAt,omitempty"`
}

type ContainerStateTerminated struct {
	ExitCode   int       `json:"exitCode" yaml:"exitCode"`
	Signal     int       `json:"signal,omitempty" yaml:"signal,omitempty"`
	Reason     string    `json:"reason,omitempty" yaml:"reason,omitempty"`
	Message    string    `json:"message,omitempty" yaml:"message,omitempty"`
	StartedAt  time.Time `json:"startedAt,omitempty" yaml:"startedAt,omitempty"`
	FinishedAt time.Time `json:"finishedAt,omitempty" yaml:"finishedAt,omitempty"`
}

type ContainerState struct {
	// Only one of the following ContainerState may be specified.
	// If none of them is specified, the default one is ContainerStateWaiting.
	Waiting     *ContainerStateWaiting    `json:"waiting,omitempty" yaml:"waiting,omitempty"`
	Running     *ContainerStateRunning    `json:"running,omitempty" yaml:"running,omitempty"`
	Termination *ContainerStateTerminated `json:"termination,omitempty" yaml:"termination,omitempty"`
}

type ContainerStatus struct {
	// TODO(dchen1107): Should we rename PodStatus to a more generic name or have a separate states
	// defined for container?
	State ContainerState `json:"state,omitempty" yaml:"state,omitempty"`
	// Note that this is calculated from dead containers.  But those containers are subject to
	// garbage collection.  This value will get capped at 5 by GC.
	RestartCount int `json:"restartCount" yaml:"restartCount"`
	// TODO(dchen1107): Deprecated this soon once we pull entire PodStatus from node,
	// not just PodInfo. Now we need this to remove docker.Container from API
	PodIP string `json:"podIP,omitempty" yaml:"podIP,omitempty"`
	// TODO(dchen1107): Need to decide how to reprensent this in v1beta3
	Image string `yaml:"image" json:"image"`
	// TODO(dchen1107): Once we have done with integration with cadvisor, resource
	// usage should be included.
}

// PodInfo contains one entry for every container with available info.
type PodInfo map[string]ContainerStatus

type RestartPolicyAlways struct{}

// TODO(dchen1107): Define what kinds of failures should restart.
// TODO(dchen1107): Decide whether to support policy knobs, and, if so, which ones.
type RestartPolicyOnFailure struct{}

type RestartPolicyNever struct{}

type RestartPolicy struct {
	// Only one of the following restart policies may be specified.
	// If none of the following policies is specified, the default one
	// is RestartPolicyAlways.
	Always    *RestartPolicyAlways    `json:"always,omitempty" yaml:"always,omitempty"`
	OnFailure *RestartPolicyOnFailure `json:"onFailure,omitempty" yaml:"onFailure,omitempty"`
	Never     *RestartPolicyNever     `json:"never,omitempty" yaml:"never,omitempty"`
}

// PodState is the state of a pod, used as either input (desired state) or output (current state).
type PodState struct {
	Manifest ContainerManifest `json:"manifest,omitempty" yaml:"manifest,omitempty"`
	Status   PodStatus         `json:"status,omitempty" yaml:"status,omitempty"`
	Host     string            `json:"host,omitempty" yaml:"host,omitempty"`
	HostIP   string            `json:"hostIP,omitempty" yaml:"hostIP,omitempty"`
	PodIP    string            `json:"podIP,omitempty" yaml:"podIP,omitempty"`

	// The key of this map is the *name* of the container within the manifest; it has one
	// entry per container in the manifest. The value of this map is currently the output
	// of `docker inspect`. This output format is *not* final and should not be relied
	// upon.
	// TODO: Make real decisions about what our info should look like. Re-enable fuzz test
	// when we have done this.
	Info PodInfo `json:"info,omitempty" yaml:"info,omitempty"`
}

// PodList is a list of Pods.
type PodList struct {
	TypeMeta `json:",inline" yaml:",inline"`
	Items    []Pod `json:"items" yaml:"items"`
}

// Pod is a collection of containers, used as either input (create, update) or as output (list, get).
type Pod struct {
	TypeMeta     `json:",inline" yaml:",inline"`
	Labels       map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
	DesiredState PodState          `json:"desiredState,omitempty" yaml:"desiredState,omitempty"`
	CurrentState PodState          `json:"currentState,omitempty" yaml:"currentState,omitempty"`
	// NodeSelector is a selector which must be true for the pod to fit on a node
	NodeSelector map[string]string `json:"nodeSelector,omitempty" yaml:"nodeSelector,omitempty"`
}

// ReplicationControllerState is the state of a replication controller, either input (create, update) or as output (list, get).
type ReplicationControllerState struct {
	Replicas        int               `json:"replicas" yaml:"replicas"`
	ReplicaSelector map[string]string `json:"replicaSelector,omitempty" yaml:"replicaSelector,omitempty"`
	PodTemplate     PodTemplate       `json:"podTemplate,omitempty" yaml:"podTemplate,omitempty"`
}

// ReplicationControllerList is a collection of replication controllers.
type ReplicationControllerList struct {
	TypeMeta `json:",inline" yaml:",inline"`
	Items    []ReplicationController `json:"items" yaml:"items"`
}

// ReplicationController represents the configuration of a replication controller.
type ReplicationController struct {
	TypeMeta     `json:",inline" yaml:",inline"`
	DesiredState ReplicationControllerState `json:"desiredState,omitempty" yaml:"desiredState,omitempty"`
	CurrentState ReplicationControllerState `json:"currentState,omitempty" yaml:"currentState,omitempty"`
	Labels       map[string]string          `json:"labels,omitempty" yaml:"labels,omitempty"`
}

// PodTemplate holds the information used for creating pods.
type PodTemplate struct {
	DesiredState PodState          `json:"desiredState,omitempty" yaml:"desiredState,omitempty"`
	Labels       map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}

// ServiceList holds a list of services.
type ServiceList struct {
	TypeMeta `json:",inline" yaml:",inline"`
	Items    []Service `json:"items" yaml:"items"`
}

// Service is a named abstraction of software service (for example, mysql) consisting of local port
// (for example 3306) that the proxy listens on, and the selector that determines which pods
// will answer requests sent through the proxy.
type Service struct {
	TypeMeta `json:",inline" yaml:",inline"`

	// Required.
	Port int `json:"port" yaml:"port"`
	// Optional: Defaults to "TCP".
	Protocol Protocol `yaml:"protocol,omitempty" json:"protocol,omitempty"`

	// This service's labels.
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`

	// This service will route traffic to pods having labels matching this selector.
	Selector                   map[string]string `json:"selector,omitempty" yaml:"selector,omitempty"`
	CreateExternalLoadBalancer bool              `json:"createExternalLoadBalancer,omitempty" yaml:"createExternalLoadBalancer,omitempty"`

	// ContainerPort is the name of the port on the container to direct traffic to.
	// Optional, if unspecified use the first port on the container.
	ContainerPort util.IntOrString `json:"containerPort,omitempty" yaml:"containerPort,omitempty"`

	// PortalIP is usually assigned by the master.  If specified by the user
	// we will try to respect it or else fail the request.  This field can
	// not be changed by updates.
	PortalIP string `json:"portalIP,omitempty" yaml:"portalIP,omitempty"`

	// ProxyPort is assigned by the master.  If specified by the user it will be ignored.
	ProxyPort int `json:"proxyPort,omitempty" yaml:"proxyPort,omitempty"`
}

// Endpoints is a collection of endpoints that implement the actual service, for example:
// Name: "mysql", Endpoints: ["10.10.1.1:1909", "10.10.2.2:8834"]
type Endpoints struct {
	TypeMeta  `json:",inline" yaml:",inline"`
	Endpoints []string `json:"endpoints,omitempty" yaml:"endpoints,omitempty"`
}

// EndpointsList is a list of endpoints.
type EndpointsList struct {
	TypeMeta `json:",inline" yaml:",inline"`
	Items    []Endpoints `json:"items" yaml:"items"`
}

// NodeResources represents resources on a Kubernetes system node
// see https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/resources.md for more details.
type NodeResources struct {
	// Capacity represents the available resources.
	Capacity ResourceList `json:"capacity,omitempty" yaml:"capacity,omitempty"`
}

type ResourceName string

type ResourceList map[ResourceName]util.IntOrString

// Minion is a worker node in Kubernetenes.
// The name of the minion according to etcd is in ID.
type Minion struct {
	TypeMeta `json:",inline" yaml:",inline"`
	// Queried from cloud provider, if available.
	HostIP string `json:"hostIP,omitempty" yaml:"hostIP,omitempty"`
	// Resources available on the node
	NodeResources NodeResources `json:"resources,omitempty" yaml:"resources,omitempty"`
	// Labels for the node
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}

// MinionList is a list of minions.
type MinionList struct {
	TypeMeta `json:",inline" yaml:",inline"`
	Items    []Minion `json:"items" yaml:"items"`
}

// Binding is written by a scheduler to cause a pod to be bound to a host.
type Binding struct {
	TypeMeta `json:",inline" yaml:",inline"`
	PodID    string `json:"podID" yaml:"podID"`
	Host     string `json:"host" yaml:"host"`
}

// Status is a return value for calls that don't return other objects.
// TODO: this could go in apiserver, but I'm including it here so clients needn't
// import both.
type Status struct {
	TypeMeta `json:",inline" yaml:",inline"`
	// One of: "Success", "Failure", "Working" (for operations not yet completed)
	Status string `json:"status,omitempty" yaml:"status,omitempty"`
	// A human-readable description of the status of this operation.
	Message string `json:"message,omitempty" yaml:"message,omitempty"`
	// A machine-readable description of why this operation is in the
	// "Failure" or "Working" status. If this value is empty there
	// is no information available. A Reason clarifies an HTTP status
	// code but does not override it.
	Reason StatusReason `json:"reason,omitempty" yaml:"reason,omitempty"`
	// Extended data associated with the reason.  Each reason may define its
	// own extended details. This field is optional and the data returned
	// is not guaranteed to conform to any schema except that defined by
	// the reason type.
	Details *StatusDetails `json:"details,omitempty" yaml:"details,omitempty"`
	// Suggested HTTP return code for this status, 0 if not set.
	Code int `json:"code,omitempty" yaml:"code,omitempty"`
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
	ID string `json:"id,omitempty" yaml:"id,omitempty"`
	// The kind attribute of the resource associated with the status StatusReason.
	// On some operations may differ from the requested resource Kind.
	Kind string `json:"kind,omitempty" yaml:"kind,omitempty"`
	// The Causes array includes more details associated with the StatusReason
	// failure. Not all StatusReasons may provide detailed causes.
	Causes []StatusCause `json:"causes,omitempty" yaml:"causes,omitempty"`
}

// Values of Status.Status
const (
	StatusSuccess = "Success"
	StatusFailure = "Failure"
	StatusWorking = "Working"
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

	// StatusReasonWorking means the server is processing this request and will complete
	// at a future time.
	// Details (optional):
	//   "kind" string - the name of the resource being referenced ("operation" today)
	//   "id"   string - the identifier of the Operation resource where updates
	//                   will be returned
	// Headers (optional):
	//   "Location" - HTTP header populated with a URL that can retrieved the final
	//                status of this operation.
	// Status code 202
	StatusReasonWorking StatusReason = "Working"

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
)

// StatusCause provides more information about an api.Status failure, including
// cases when multiple errors are encountered.
type StatusCause struct {
	// A machine-readable description of the cause of the error. If this value is
	// empty there is no information available.
	Type CauseType `json:"reason,omitempty" yaml:"reason,omitempty"`
	// A human-readable description of the cause of the error.  This field may be
	// presented as-is to a reader.
	Message string `json:"message,omitempty" yaml:"message,omitempty"`
	// The field of the resource that has caused this error, as named by its JSON
	// serialization. May include dot and postfix notation for nested attributes.
	// Arrays are zero-indexed.  Fields may appear more than once in an array of
	// causes due to fields having multiple errors.
	// Optional.
	//
	// Examples:
	//   "name" - the field "name" on the current resource
	//   "items[0].name" - the field "name" on the first array entry in "items"
	Field string `json:"field,omitempty" yaml:"field,omitempty"`
}

// CauseType is a machine readable value providing more detail about what
// occured in a status response. An operation may have multiple causes for a
// status (whether Failure, Success, or Working).
type CauseType string

const (
	// CauseTypeFieldValueNotFound is used to report failure to find a requested value
	// (e.g. looking up an ID).
	CauseTypeFieldValueNotFound CauseType = "FieldValueNotFound"
	// CauseTypeFieldValueInvalid is used to report required values that are not
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

// ServerOp is an operation delivered to API clients.
type ServerOp struct {
	TypeMeta `yaml:",inline" json:",inline"`
}

// ServerOpList is a list of operations, as delivered to API clients.
type ServerOpList struct {
	TypeMeta `yaml:",inline" json:",inline"`
	Items    []ServerOp `yaml:"items" json:"items"`
}

// ObjectReference contains enough information to let you inspect or modify the referred object.
type ObjectReference struct {
	Kind            string `json:"kind,omitempty" yaml:"kind,omitempty"`
	Namespace       string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	ID              string `json:"name,omitempty" yaml:"name,omitempty"`
	UID             string `json:"uid,omitempty" yaml:"uid,omitempty"`
	APIVersion      string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	ResourceVersion string `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`

	// Optional. If referring to a piece of an object instead of an entire object, this string
	// should contain a valid field access statement. For example,
	// if the object reference is to a container within a pod, this would take on a value like:
	// "desiredState.manifest.containers[2]". Such statements are valid language constructs in
	// both go and JavaScript. This is syntax is chosen only to have some well-defined way of
	// referencing a part of an object.
	// TODO: this design is not final and this field is subject to change in the future.
	FieldPath string `json:"fieldPath,omitempty" yaml:"fieldPath,omitempty"`
}

// Event is a report of an event somewhere in the cluster.
// TODO: Decide whether to store these separately or with the object they apply to.
type Event struct {
	TypeMeta `yaml:",inline" json:",inline"`

	// Required. The object that this event is about.
	InvolvedObject ObjectReference `json:"involvedObject,omitempty" yaml:"involvedObject,omitempty"`

	// Should be a short, machine understandable string that describes the current status
	// of the referred object. This should not give the reason for being in this state.
	// Examples: "running", "cantStart", "cantSchedule", "deleted".
	// It's OK for components to make up statuses to report here, but the same string should
	// always be used for the same status.
	// TODO: define a way of making sure these are consistent and don't collide.
	// TODO: provide exact specification for format.
	Status string `json:"status,omitempty" yaml:"status,omitempty"`

	// Optional; this should be a short, machine understandable string that gives the reason
	// for the transition into the object's current status. For example, if ObjectStatus is
	// "cantStart", StatusReason might be "imageNotFound".
	// TODO: provide exact specification for format.
	Reason string `json:"reason,omitempty" yaml:"reason,omitempty"`

	// Optional. A human-readable description of the status of this operation.
	// TODO: decide on maximum length.
	Message string `json:"message,omitempty" yaml:"message,omitempty"`

	// Optional. The component reporting this event. Should be a short machine understandable string.
	// TODO: provide exact specification for format.
	Source string `json:"source,omitempty" yaml:"source,omitempty"`

	// The time at which the client recorded the event. (Time of server receipt is in TypeMeta.)
	Timestamp util.Time `json:"timestamp,omitempty" yaml:"timestamp,omitempty"`
}

// EventList is a list of events.
type EventList struct {
	TypeMeta `yaml:",inline" json:",inline"`
	Items    []Event `yaml:"items" json:"items"`
}

// ContainerManifest corresponds to the Container Manifest format, documented at:
// https://developers.google.com/compute/docs/containers/container_vms#container_manifest
// This is used as the representation of Kubernetes workloads.
// DEPRECATED: Replaced with BoundPod
type ContainerManifest struct {
	// Required: This must be a supported version string, such as "v1beta1".
	Version string `yaml:"version" json:"version"`
	// Required: This must be a DNS_SUBDOMAIN.
	// TODO: ID on Manifest is deprecated and will be removed in the future.
	ID string `yaml:"id" json:"id"`
	// TODO: UUID on Manifest is deprecated in the future once we are done
	// with the API refactoring. It is required for now to determine the instance
	// of a Pod.
	UUID          string        `yaml:"uuid,omitempty" json:"uuid,omitempty"`
	Volumes       []Volume      `yaml:"volumes" json:"volumes"`
	Containers    []Container   `yaml:"containers" json:"containers"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty" yaml:"restartPolicy,omitempty"`
}

// ContainerManifestList is used to communicate container manifests to kubelet.
// DEPRECATED: Replaced with BoundPods
type ContainerManifestList struct {
	TypeMeta `json:",inline" yaml:",inline"`
	Items    []ContainerManifest `json:"items" yaml:"items,omitempty"`
}

// Backported from v1beta3 to replace ContainerManifest

// PodSpec is a description of a pod
type PodSpec struct {
	Volumes       []Volume      `json:"volumes" yaml:"volumes"`
	Containers    []Container   `json:"containers" yaml:"containers"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty" yaml:"restartPolicy,omitempty"`
}

// BoundPod is a collection of containers that should be run on a host. A BoundPod
// defines how a Pod may change after a Binding is created. A Pod is a request to
// execute a pod, whereas a BoundPod is the specification that would be run on a server.
type BoundPod struct {
	TypeMeta `json:",inline" yaml:",inline"`

	// Spec defines the behavior of a pod.
	Spec PodSpec `json:"spec,omitempty" yaml:"spec,omitempty"`
}

// BoundPods is a list of Pods bound to a common server. The resource version of
// the pod list is guaranteed to only change when the list of bound pods changes.
type BoundPods struct {
	TypeMeta `json:",inline" yaml:",inline"`

	// Host is the name of a node that these pods were bound to.
	Host string `json:"host" yaml:"host"`

	// Items is the list of all pods bound to a given host.
	Items []BoundPod `json:"items" yaml:"items"`
}
