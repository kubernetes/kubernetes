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

package v1beta1

import (
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

// ContainerManifest corresponds to the Container Manifest format, documented at:
// https://developers.google.com/compute/docs/containers/container_vms#container_manifest
// This is used as the representation of Kubernetes workloads.
type ContainerManifest struct {
	// Required: This must be a supported version string, such as "v1beta1".
	Version string `json:"version" description:"manifest version; must be v1beta1"`
	// Required: This must be a DNS_SUBDOMAIN.
	// TODO: ID on Manifest is deprecated and will be removed in the future.
	ID string `json:"id" description:"manifest name; must be a DNS_SUBDOMAIN"`
	// TODO: UUID on Manifext is deprecated in the future once we are done
	// with the API refactory. It is required for now to determine the instance
	// of a Pod.
	UUID          types.UID     `json:"uuid,omitempty" description:"manifest UUID"`
	Volumes       []Volume      `json:"volumes" description:"list of volumes that can be mounted by containers belonging to the pod"`
	Containers    []Container   `json:"containers" description:"list of containers belonging to the pod"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty" description:"restart policy for all containers within the pod; one of RestartPolicyAlways, RestartPolicyOnFailure, RestartPolicyNever"`
	// Optional: Set DNS policy.  Defaults to "ClusterFirst"
	DNSPolicy DNSPolicy `json:"dnsPolicy,omitempty" description:"DNS policy for containers within the pod; one of 'ClusterFirst' or 'Default'"`
}

// ContainerManifestList is used to communicate container manifests to kubelet.
type ContainerManifestList struct {
	TypeMeta `json:",inline"`
	Items    []ContainerManifest `json:"items" description:"list of pod container manifests"`
}

// Volume represents a named volume in a pod that may be accessed by any containers in the pod.
type Volume struct {
	// Required: This must be a DNS_LABEL.  Each volume in a pod must have
	// a unique name.
	Name string `json:"name" description:"volume name; must be a DNS_LABEL and unique within the pod"`
	// Source represents the location and type of a volume to mount.
	// This is optional for now. If not specified, the Volume is implied to be an EmptyDir.
	// This implied behavior is deprecated and will be removed in a future version.
	Source *VolumeSource `json:"source" description:"location and type of volume to mount; at most one of HostDir, EmptyDir, GCEPersistentDisk, or GitRepo; default is EmptyDir"`
}

// VolumeSource represents the source location of a valume to mount.
// Only one of its members may be specified.
type VolumeSource struct {
	// HostDir represents a pre-existing directory on the host machine that is directly
	// exposed to the container. This is generally used for system agents or other privileged
	// things that are allowed to see the host machine. Most containers will NOT need this.
	// TODO(jonesdl) We need to restrict who can use host directory mounts and
	// who can/can not mount host directories as read/write.
	HostDir *HostPath `json:"hostDir" description:"pre-existing host file or directory; generally for privileged system daemons or other agents tied to the host"`
	// EmptyDir represents a temporary directory that shares a pod's lifetime.
	EmptyDir *EmptyDir `json:"emptyDir" description:"temporary directory that shares a pod's lifetime"`
	// GCEPersistentDisk represents a GCE Disk resource that is attached to a
	// kubelet's host machine and then exposed to the pod.
	GCEPersistentDisk *GCEPersistentDisk `json:"persistentDisk" description:"GCE disk resource attached to the host machine on demand"`
	// GitRepo represents a git repository at a particular revision.
	GitRepo *GitRepo `json:"gitRepo" description:"git repository at a particular revision"`
}

// HostPath represents bare host directory volume.
type HostPath struct {
	Path string `json:"path" description:"path of the directory on the host"`
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

// GCEPersistentDisk represents a Persistent Disk resource in Google Compute Engine.
//
// A GCE PD must exist and be formatted before mounting to a container.
// The disk must also be in the same GCE project and zone as the kubelet.
// A GCE PD can only be mounted as read/write once.
type GCEPersistentDisk struct {
	// Unique name of the PD resource. Used to identify the disk in GCE
	PDName string `json:"pdName" description:"unique name of the PD resource in GCE"`
	// Required: Filesystem type to mount.
	// Must be a filesystem type supported by the host operating system.
	// Ex. "ext4", "xfs", "ntfs"
	// TODO: how do we prevent errors in the filesystem from compromising the machine
	// TODO: why omitempty if required?
	FSType string `json:"fsType,omitempty" description:"file system type to mount, such as ext4, xfs, ntfs"`
	// Optional: Partition on the disk to mount.
	// If omitted, kubelet will attempt to mount the device name.
	// Ex. For /dev/sda1, this field is "1", for /dev/sda, this field 0 or empty.
	Partition int `json:"partition,omitempty" description:"partition on the disk to mount (e.g., '1' for /dev/sda1); if omitted the plain device name (e.g., /dev/sda) will be mounted"`
	// Optional: Defaults to false (read/write). ReadOnly here will force
	// the ReadOnly setting in VolumeMounts.
	ReadOnly bool `json:"readOnly,omitempty" description:"read-only if true, read-write otherwise (false or unspecified)"`
}

// GitRepo represents a volume that is pulled from git when the pod is created.
type GitRepo struct {
	// Repository URL
	Repository string `json:"repository" description:"repository URL"`
	// Commit hash, this is optional
	Revision string `json:"revision" description:"commit hash for the specified revision"`
}

// Port represents a network port in a single container
type Port struct {
	// Optional: If specified, this must be a DNS_LABEL.  Each named port
	// in a pod must have a unique name.
	Name string `json:"name,omitempty" description:"name for the port that can be referred to by services; must be a DNS_LABEL and unique without the pod"`
	// Optional: If specified, this must be a valid port number, 0 < x < 65536.
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
	// Exactly one of the following must be set.  If both are set, prefer MountPath.
	// DEPRECATED: Path will be removed in a future version of the API.
	MountPath string `json:"mountPath" description:"path within the container at which the volume should be mounted; overrides path"`
	Path      string `json:"path,omitempty" description:"path within the container at which the volume should be mounted; deprecated"`
	// One of: "LOCAL" (local volume) or "HOST" (external mount from the host). Default: LOCAL.
	// DEPRECATED: MountType will be removed in a future version of the API.
	MountType string `json:"mountType,omitempty" description:"LOCAL or HOST; defaults to LOCAL; deprecated"`
}

// EnvVar represents an environment variable present in a Container.
type EnvVar struct {
	// Required: This must be a C_IDENTIFIER.
	// Exactly one of the following must be set.  If both are set, prefer Name.
	// DEPRECATED: EnvVar.Key will be removed in a future version of the API.
	Name string `json:"name" description:"name of the environment variable; must be a C_IDENTIFIER"`
	Key  string `json:"key,omitempty" description:"name of the environment variable; must be a C_IDENTIFIER; deprecated - use name instead"`
	// Optional: defaults to "".
	Value string `json:"value,omitempty" description:"value of the environment variable; defaults to empty string"`
}

// HTTPGetAction describes an action based on HTTP Get requests.
type HTTPGetAction struct {
	// Optional: Path to access on the HTTP server.
	Path string `json:"path,omitempty" description:"path to access on the HTTP server"`
	// Required: Name or number of the port to access on the container.
	Port util.IntOrString `json:"port,omitempty" description:"number or name of the port to access on the container"`
	// Optional: Host name to connect to, defaults to the pod IP.
	Host string `json:"host,omitempty" description:"hostname to connect to; defaults to pod IP"`
}

// TCPSocketAction describes an action based on opening a socket
type TCPSocketAction struct {
	// Required: Port to connect to.
	Port util.IntOrString `json:"port,omitempty" description:"number of name of the port to access on the container"`
}

// ExecAction describes a "run in container" action.
type ExecAction struct {
	// Command is the command line to execute inside the container, the working directory for the
	// command  is root ('/') in the container's filesystem.  The command is simply exec'd, it is
	// not run inside a shell, so traditional shell instructions ('|', etc) won't work.  To use
	// a shell, you need to explicitly call out to that shell.
	// A return code of zero is treated as 'Healthy', non-zero is 'Unhealthy'
	Command []string `json:"command,omitempty" description:"command line to execute inside the container; working directory for the command is root ('/') in the container's file system; the command is exec'd, not run inside a shell; exit status of 0 is treated as live/healthy and non-zero is unhealthy"`
}

// LivenessProbe describes a liveness probe to be examined to the container.
// TODO: pass structured data to the actions, and document that data here.
type LivenessProbe struct {
	// HTTPGetProbe parameters, required if Type == 'http'
	HTTPGet *HTTPGetAction `json:"httpGet,omitempty" description:"parameters for HTTP-based liveness probe"`
	// TCPSocketProbe parameter, required if Type == 'tcp'
	TCPSocket *TCPSocketAction `json:"tcpSocket,omitempty" description:"parameters for TCP-based liveness probe"`
	// ExecProbe parameter, required if Type == 'exec'
	Exec *ExecAction `json:"exec,omitempty" description:"parameters for exec-based liveness probe"`
	// Length of time before health checking is activated.  In seconds.
	InitialDelaySeconds int64 `json:"initialDelaySeconds,omitempty" description:"number of seconds after the container has started before liveness probes are initiated"`
}

// PullPolicy describes a policy for if/when to pull a container image
type PullPolicy string

const (
	// PullAlways means that kubelet always attempts to pull the latest image.  Container will fail If the pull fails.
	PullAlways PullPolicy = "PullAlways"
	// PullNever means that kubelet never pulls an image, but only uses a local image.  Container will fail if the image isn't present
	PullNever PullPolicy = "PullNever"
	// PullIfNotPresent means that kubelet pulls if the image isn't present on disk. Container will fail if the image isn't present and the pull fails.
	PullIfNotPresent PullPolicy = "PullIfNotPresent"
)

// Container represents a single container that is expected to be run on the host.
type Container struct {
	// Required: This must be a DNS_LABEL.  Each container in a pod must
	// have a unique name.
	Name string `json:"name" description:"name of the container; must be a DNS_LABEL and unique within the pod"`
	// Required.
	Image string `json:"image" description:"Docker image name"`
	// Optional: Defaults to whatever is defined in the image.
	Command []string `json:"command,omitempty" description:"command argv array; not executed within a shell; defaults to entrypoint or command in the image"`
	// Optional: Defaults to Docker's default.
	WorkingDir string   `json:"workingDir,omitempty" description:"container's working directory; defaults to image's default"`
	Ports      []Port   `json:"ports,omitempty" description:"list of ports to expose from the container"`
	Env        []EnvVar `json:"env,omitempty" description:"list of environment variables to set in the container"`
	// Optional: Defaults to unlimited.
	Memory int64 `json:"memory,omitempty" description:"memory limit in bytes; defaults to unlimited"`
	// Optional: Defaults to unlimited.
	CPU           int            `json:"cpu,omitempty" description:"CPU share in thousandths of a core"`
	VolumeMounts  []VolumeMount  `json:"volumeMounts,omitempty" description:"pod volumes to mount into the container's filesystem"`
	LivenessProbe *LivenessProbe `json:"livenessProbe,omitempty" description:"periodic probe of container liveness; container will be restarted if the probe fails"`
	Lifecycle     *Lifecycle     `json:"lifecycle,omitempty" description:"actions that the management system should take in response to container lifecycle events"`
	// Optional: Defaults to /dev/termination-log
	TerminationMessagePath string `json:"terminationMessagePath,omitempty" description:"path at which the file to which the container's termination message will be written is mounted into the container's filesystem; message written is intended to be brief final status, such as an assertion failure message; defaults to /dev/termination-log"`
	// Optional: Default to false.
	Privileged bool `json:"privileged,omitempty" description:"whether or not the container is granted privileged status; defaults to false"`
	// Optional: Policy for pulling images for this container
	ImagePullPolicy PullPolicy `json:"imagePullPolicy" description:"image pull policy; one of PullAlways, PullNever, PullIfNotPresent; defaults to PullAlways if :latest tag is specified, or PullIfNotPresent otherwise"`
}

// Handler defines a specific action that should be taken
// TODO: merge this with liveness probing?
// TODO: pass structured data to these actions, and document that data here.
type Handler struct {
	// One and only one of the following should be specified.
	// Exec specifies the action to take.
	Exec *ExecAction `json:"exec,omitempty" description:"exec-based hook handler"`
	// HTTPGet specifies the http request to perform.
	HTTPGet *HTTPGetAction `json:"httpGet,omitempty" description:"HTTP-based hook handler"`
}

// Lifecycle describes actions that the management system should take in response to container lifecycle
// events.  For the PostStart and PreStop lifecycle handlers, management of the container blocks
// until the action is complete, unless the container process fails, in which case the handler is aborted.
type Lifecycle struct {
	// PostStart is called immediately after a container is created.  If the handler fails, the container
	// is terminated and restarted.
	PostStart *Handler `json:"postStart,omitempty" description:"called immediately after a container is started; if the handler fails, the container is terminated and restarted according to its restart policy; other management of the container blocks until the hook completes"`
	// PreStop is called immediately before a container is terminated.  The reason for termination is
	// passed to the handler.  Regardless of the outcome of the handler, the container is eventually terminated.
	PreStop *Handler `json:"preStop,omitempty" description:"called before a container is terminated; the container is terminated after the handler completes; other management of the container blocks until the hook completes"`
}

// The below types are used by kube_client and api_server.

// TypeMeta is shared by all objects sent to, or returned from the client.
type TypeMeta struct {
	Kind              string    `json:"kind,omitempty" description:"kind of object, in CamelCase"`
	ID                string    `json:"id,omitempty" description:"name of the object; must be a DNS_SUBDOMAIN and unique among all objects of the same kind within the same namespace; used in resource URLs"`
	UID               types.UID `json:"uid,omitempty" description:"UUID assigned by the system upon creation, unique across space and time"`
	CreationTimestamp util.Time `json:"creationTimestamp,omitempty" description:"RFC 3339 date and time at which the object was created; recorded by the system; null for lists"`
	SelfLink          string    `json:"selfLink,omitempty" description:"URL for the object"`
	ResourceVersion   uint64    `json:"resourceVersion,omitempty" description:"string that identifies the internal version of this object that can be used by clients to determine when objects have changed; value must be treated as opaque by clients and passed unmodified back to the server"`
	APIVersion        string    `json:"apiVersion,omitempty" description:"version of the schema the object should have"`
	Namespace         string    `json:"namespace,omitempty" description:"namespace to which the object belongs; must be a DNS_SUBDOMAIN; 'default' by default"`

	// Annotations are unstructured key value data stored with a resource that may be set by
	// external tooling. They are not queryable and should be preserved when modifying
	// objects.
	Annotations map[string]string `json:"annotations,omitempty" description:"map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about the object"`
}

// PodStatus represents a status of a pod.
type PodStatus string

// These are the valid statuses of pods.
const (
	// PodWaiting means that we're waiting for the pod to begin running.
	PodWaiting PodStatus = "Waiting"
	// PodRunning means that the pod is up and running.
	PodRunning PodStatus = "Running"
	// PodTerminated means that the pod has stopped with error(s)
	PodTerminated PodStatus = "Terminated"
	// PodUnknown means that we failed to obtain info about the pod.
	PodUnknown PodStatus = "Unknown"
	// PodSucceeded means that the pod has stopped without error(s)
	PodSucceeded PodStatus = "Succeeded"
)

type ContainerStateWaiting struct {
	// Reason could be pulling image,
	Reason string `json:"reason,omitempty" description:"(brief) reason the container is not yet running, such as pulling its image"`
}

type ContainerStateRunning struct {
	StartedAt util.Time `json:"startedAt,omitempty" description:"time at which the container was last (re-)started"`
}

type ContainerStateTerminated struct {
	ExitCode   int       `json:"exitCode" description:"exit status from the last termination of the container"`
	Signal     int       `json:"signal,omitempty" description:"signal from the last termination of the container"`
	Reason     string    `json:"reason,omitempty" description:"(brief) reason from the last termination of the container"`
	Message    string    `json:"message,omitempty" description:"message regarding the last termination of the container"`
	StartedAt  util.Time `json:"startedAt,omitempty" description:"time at which previous execution of the container started"`
	FinishedAt util.Time `json:"finishedAt,omitempty" description:"time at which the container last terminated"`
}

// ContainerState holds a possible state of container.
// Only one of its members may be specified.
// If none of them is specified, the default one is ContainerStateWaiting.
type ContainerState struct {
	Waiting     *ContainerStateWaiting    `json:"waiting,omitempty" description:"details about a waiting container"`
	Running     *ContainerStateRunning    `json:"running,omitempty" description:"details about a running container"`
	Termination *ContainerStateTerminated `json:"termination,omitempty" description:"details about a terminated container"`
}

type ContainerStatus struct {
	// TODO(dchen1107): Should we rename PodStatus to a more generic name or have a separate states
	// defined for container?
	State ContainerState `json:"state,omitempty" description:"details about the container's current condition"`
	// Note that this is calculated from dead containers.  But those containers are subject to
	// garbage collection.  This value will get capped at 5 by GC.
	RestartCount int `json:"restartCount" description:"the number of times the container has been restarted, currently based on the number of dead containers that have not yet been removed"`
	// TODO(dchen1107): Deprecated this soon once we pull entire PodStatus from node,
	// not just PodInfo. Now we need this to remove docker.Container from API
	PodIP string `json:"podIP,omitempty" description:"pod's IP address"`
	// TODO(dchen1107): Need to decide how to reprensent this in v1beta3
	Image       string `json:"image" description:"image of the container"`
	ContainerID string `json:"containerID,omitempty" description:"container's ID in the format 'docker://<container_id>'"`
}

// PodInfo contains one entry for every container with available info.
type PodInfo map[string]ContainerStatus

// PodContainerInfo is a wrapper for PodInfo that can be encode/decoded
type PodContainerInfo struct {
	TypeMeta      `json:",inline"`
	ContainerInfo PodInfo `json:"containerInfo" description:"information about each container in this pod"`
}

type RestartPolicyAlways struct{}

// TODO(dchen1107): Define what kinds of failures should restart
// TODO(dchen1107): Decide whether to support policy knobs, and, if so, which ones.
type RestartPolicyOnFailure struct{}

type RestartPolicyNever struct{}

type RestartPolicy struct {
	// Only one of the following restart policy may be specified.
	// If none of the following policies is specified, the default one
	// is RestartPolicyAlways.
	Always    *RestartPolicyAlways    `json:"always,omitempty" description:"always restart the container after termination"`
	OnFailure *RestartPolicyOnFailure `json:"onFailure,omitempty" description:"restart the container if it fails for any reason, but not if it succeeds (exit 0)"`
	Never     *RestartPolicyNever     `json:"never,omitempty" description:"never restart the container"`
}

// PodState is the state of a pod, used as either input (desired state) or output (current state).
type PodState struct {
	Manifest ContainerManifest `json:"manifest,omitempty" description:"manifest of containers and volumes comprising the pod"`
	Status   PodStatus         `json:"status,omitempty" description:"current condition of the pod, Waiting, Running, or Terminated"`
	// A human readable message indicating details about why the pod is in this state.
	Message string `json:"message,omitempty" description:"human readable message indicating details about why the pod is in this condition"`
	Host    string `json:"host,omitempty" description:"host to which the pod is assigned; empty if not yet scheduled"`
	HostIP  string `json:"hostIP,omitempty" description:"IP address of the host to which the pod is assigned; empty if not yet scheduled"`
	PodIP   string `json:"podIP,omitempty" description:"IP address allocated to the pod; routable at least within the cluster; empty if not yet allocated"`

	// The key of this map is the *name* of the container within the manifest; it has one
	// entry per container in the manifest. The value of this map is ContainerStatus for
	// the container.
	Info PodInfo `json:"info,omitempty" description:"map of container name to container status"`
}

type PodStatusResult struct {
	TypeMeta `json:",inline"`
	State    PodState `json:"state,omitempty" description:"current state of the pod"`
}

// PodList is a list of Pods.
type PodList struct {
	TypeMeta `json:",inline"`
	Items    []Pod `json:"items" description:"list of pods"`
}

// Pod is a collection of containers, used as either input (create, update) or as output (list, get).
type Pod struct {
	TypeMeta     `json:",inline"`
	Labels       map[string]string `json:"labels,omitempty" description:"map of string keys and values that can be used to organize and categorize pods; may match selectors of replication controllers and services"`
	DesiredState PodState          `json:"desiredState,omitempty" description:"specification of the desired state of the pod"`
	CurrentState PodState          `json:"currentState,omitempty" description:"current state of the pod"`
	// NodeSelector is a selector which must be true for the pod to fit on a node
	NodeSelector map[string]string `json:"nodeSelector,omitempty" description:"selector which must match a node's labels for the pod to be scheduled on that node"`
}

// ReplicationControllerState is the state of a replication controller, either input (create, update) or as output (list, get).
type ReplicationControllerState struct {
	Replicas        int               `json:"replicas" description:"number of replicas (desired or observed, as appropriate)"`
	ReplicaSelector map[string]string `json:"replicaSelector,omitempty" description:"label keys and values that must match in order to be controlled by this replication controller"`
	PodTemplate     PodTemplate       `json:"podTemplate,omitempty" description:"template for pods to be created by this replication controller when the observed number of replicas is less than the desired number of replicas"`
}

// ReplicationControllerList is a collection of replication controllers.
type ReplicationControllerList struct {
	TypeMeta `json:",inline"`
	Items    []ReplicationController `json:"items" description:"list of replication controllers"`
}

// ReplicationController represents the configuration of a replication controller.
type ReplicationController struct {
	TypeMeta     `json:",inline"`
	DesiredState ReplicationControllerState `json:"desiredState,omitempty" description:"specification of the desired state of the replication controller"`
	CurrentState ReplicationControllerState `json:"currentState,omitempty" description:"current state of the replication controller"`
	Labels       map[string]string          `json:"labels,omitempty" description:"map of string keys and values that can be used to organize and categorize replication controllers"`
}

// PodTemplate holds the information used for creating pods.
type PodTemplate struct {
	DesiredState PodState          `json:"desiredState,omitempty" description:"specification of the desired state of pods created from this template"`
	Labels       map[string]string `json:"labels,omitempty" description:"map of string keys and values that can be used to organize and categorize the pods created from the template; must match the selector of the replication controller to which the template belongs; may match selectors of services"`
}

// Session Affinity Type string
type AffinityType string

const (
	// AffinityTypeClientIP is the Client IP based.
	AffinityTypeClientIP AffinityType = "ClientIP"

	// AffinityTypeNone - no session affinity.
	AffinityTypeNone AffinityType = "None"
)

// ServiceList holds a list of services.
type ServiceList struct {
	TypeMeta `json:",inline"`
	Items    []Service `json:"items" description:"list of services"`
}

// Service is a named abstraction of software service (for example, mysql) consisting of local port
// (for example 3306) that the proxy listens on, and the selector that determines which pods
// will answer requests sent through the proxy.
type Service struct {
	TypeMeta `json:",inline"`

	// Required.
	Port int `json:"port" description:"port exposed by the service"`
	// Optional: Defaults to "TCP".
	Protocol Protocol `json:"protocol,omitempty" description:"protocol for port; must be UDP or TCP; TCP if unspecified"`

	// This service's labels.
	Labels map[string]string `json:"labels,omitempty" description:"map of string keys and values that can be used to organize and categorize services"`

	// This service will route traffic to pods having labels matching this selector. If null, no endpoints will be automatically created. If empty, all pods will be selected.
	Selector map[string]string `json:"selector" description:"label keys and values that must match in order to receive traffic for this service; if empty, all pods are selected, if not specified, endpoints must be manually specified"`
	// An external load balancer should be set up via the cloud-provider
	CreateExternalLoadBalancer bool `json:"createExternalLoadBalancer,omitempty" description:"set up a cloud-provider-specific load balancer on an external IP"`

	// PublicIPs are used by external load balancers.
	PublicIPs []string `json:"publicIPs,omitempty" description:"externally visible IPs from which to select the address for the external load balancer"`

	// ContainerPort is the name of the port on the container to direct traffic to.
	// Optional, if unspecified use the first port on the container.
	ContainerPort util.IntOrString `json:"containerPort,omitempty" description:"number or name of the port to access on the containers belonging to pods targeted by the service"`

	// PortalIP is usually assigned by the master.  If specified by the user
	// we will try to respect it or else fail the request.  This field can
	// not be changed by updates.
	PortalIP string `json:"portalIP,omitempty" description:"IP address of the service; usually assigned by the system; if specified, it will be allocated to the service if unused, and creation of the service will fail otherwise; cannot be updated"`

	// ProxyPort is assigned by the master.  If specified by the user it will be ignored.
	ProxyPort int `json:"proxyPort,omitempty" description:"if non-zero, a pre-allocated host port used for this service by the proxy on each node; assigned by the master and ignored on input"`

	// Optional: Supports "ClientIP" and "None".  Used to maintain session affinity.
	SessionAffinity AffinityType `json:"sessionAffinity,omitempty" description:"enable client IP based session affinity; must be ClientIP or None; defaults to None"`
}

// Endpoints is a collection of endpoints that implement the actual service, for example:
// Name: "mysql", Endpoints: ["10.10.1.1:1909", "10.10.2.2:8834"]
type Endpoints struct {
	TypeMeta  `json:",inline"`
	Endpoints []string `json:"endpoints,omitempty" description:"list of endpoints corresponding to a service, of the form address:port, such as 10.10.1.1:1909"`
}

// EndpointsList is a list of endpoints.
type EndpointsList struct {
	TypeMeta `json:",inline"`
	Items    []Endpoints `json:"items" description:"list of service endpoint lists"`
}

// NodeStatus is information about the current status of a node.
type NodeStatus struct {
	// NodePhase is the current lifecycle phase of the node.
	Phase NodePhase `json:"phase,omitempty" description:"node phase is the current lifecycle phase of the node"`
	// Conditions is an array of current node conditions.
	Conditions []NodeCondition `json:"conditions,omitempty" description:"conditions is an array of current node conditions"`
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

type NodeConditionKind string

// These are valid conditions of node. Currently, we don't have enough information to decide
// node condition. In the future, we will add more. The proposed set of conditions are:
// NodeReachable, NodeLive, NodeReady, NodeSchedulable, NodeRunnable.
const (
	// NodeReachable means the node can be reached (in the sense of HTTP connection) from node controller.
	NodeReachable NodeConditionKind = "Reachable"
	// NodeReady means the node returns StatusOK for HTTP health check.
	NodeReady NodeConditionKind = "Ready"
)

type NodeConditionStatus string

// These are valid condition status. "ConditionFull" means node is in the condition;
// "ConditionNone" means node is not in the condition; "ConditionUnknown" means kubernetes
// can't decide if node is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
const (
	ConditionFull    NodeConditionStatus = "Full"
	ConditionNone    NodeConditionStatus = "None"
	ConditionUnknown NodeConditionStatus = "Unknown"
)

type NodeCondition struct {
	Kind               NodeConditionKind   `json:"kind" description:"kind of the condition, one of reachable, ready"`
	Status             NodeConditionStatus `json:"status" description:"status of the condition, one of full, none, unknown"`
	LastTransitionTime util.Time           `json:"lastTransitionTime,omitempty" description:"last time the condition transit from one status to another"`
	Reason             string              `json:"reason,omitempty" description:"(brief) reason for the condition's last transition"`
	Message            string              `json:"message,omitempty" description:"human readable message indicating details about last transition"`
}

// NodeResources represents resources on a Kubernetes system node
// see https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/resources.md for more details.
type NodeResources struct {
	// Capacity represents the available resources.
	Capacity ResourceList `json:"capacity,omitempty" description:"resource capacity of a node represented as a map of resource name to quantity of resource"`
}

type ResourceName string

const (
	// CPU, in cores. (floating point w/ 3 decimal places)
	ResourceCPU ResourceName = "cpu"
	// Memory, in bytes.
	ResourceMemory ResourceName = "memory"
)

type ResourceList map[ResourceName]util.IntOrString

// Minion is a worker node in Kubernetenes.
// The name of the minion according to etcd is in ID.
type Minion struct {
	TypeMeta `json:",inline"`
	// Queried from cloud provider, if available.
	HostIP string `json:"hostIP,omitempty" description:"IP address of the node"`
	// Resources available on the node
	NodeResources NodeResources `json:"resources,omitempty" description:"characterization of node resources"`
	// Status describes the current status of a node
	Status NodeStatus `json:"status,omitempty" description:"current status of node"`
	// Labels for the node
	Labels map[string]string `json:"labels,omitempty" description:"map of string keys and values that can be used to organize and categorize minions; labels of a minion assigned by the scheduler must match the scheduled pod's nodeSelector"`
}

// MinionList is a list of minions.
type MinionList struct {
	TypeMeta `json:",inline"`
	// DEPRECATED: the below Minions is due to a naming mistake and
	// will be replaced with Items in the future.
	Minions []Minion `json:"minions,omitempty" description:"list of nodes; deprecated"`
	Items   []Minion `json:"items" description:"list of nodes"`
}

// Binding is written by a scheduler to cause a pod to be bound to a host.
type Binding struct {
	TypeMeta `json:",inline"`
	PodID    string `json:"podID" description:"name of the pod to bind"`
	Host     string `json:"host" description:"host to which to bind the specified pod"`
}

// Status is a return value for calls that don't return other objects.
// TODO: this could go in apiserver, but I'm including it here so clients needn't
// import both.
type Status struct {
	TypeMeta `json:",inline"`
	// One of: "Success", "Failure", "Working" (for operations not yet completed)
	Status string `json:"status,omitempty" description:"status of the operation; either Working (not yet completed), Success, or Failure"`
	// A human-readable description of the status of this operation.
	Message string `json:"message,omitempty" description:"human-readable description of the status of this operation"`
	// A machine-readable description of why this operation is in the
	// "Failure" or "Working" status. If this value is empty there
	// is no information available. A Reason clarifies an HTTP status
	// code but does not override it.
	Reason StatusReason `json:"reason,omitempty" description:"machine-readable description of why this operation is in the 'Failure' or 'Working' status; if this value is empty there is no information available; a reason clarifies an HTTP status code but does not override it"`
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
	// The ID attribute of the resource associated with the status StatusReason
	// (when there is a single ID which can be described).
	ID string `json:"id,omitempty" description:"the ID attribute of the resource associated with the status StatusReason (when there is a single ID which can be described)"`
	// The kind attribute of the resource associated with the status StatusReason.
	// On some operations may differ from the requested resource Kind.
	Kind string `json:"kind,omitempty" description:"the kind attribute of the resource associated with the status StatusReason; on some operations may differ from the requested resource Kind"`
	// The Causes array includes more details associated with the StatusReason
	// failure. Not all StatusReasons may provide detailed causes.
	Causes []StatusCause `json:"causes,omitempty" description:"the Causes array includes more details associated with the StatusReason failure; not all StatusReasons may provide detailed causes"`
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
// status (whether Failure, Success, or Working).
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

// ServerOp is an operation delivered to API clients.
type ServerOp struct {
	TypeMeta `json:",inline"`
}

// ServerOpList is a list of operations, as delivered to API clients.
type ServerOpList struct {
	TypeMeta `json:",inline"`
	Items    []ServerOp `json:"items" description:"list of operations"`
}

// ObjectReference contains enough information to let you inspect or modify the referred object.
type ObjectReference struct {
	Kind            string    `json:"kind,omitempty" description:"kind of the referent"`
	Namespace       string    `json:"namespace,omitempty" description:"namespace of the referent"`
	ID              string    `json:"name,omitempty" description:"id of the referent"`
	UID             types.UID `json:"uid,omitempty" description:"uid of the referent"`
	APIVersion      string    `json:"apiVersion,omitempty" description:"API version of the referent"`
	ResourceVersion string    `json:"resourceVersion,omitempty" description:"specific resourceVersion to which this reference is made, if any"`

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

// Event is a report of an event somewhere in the cluster.
// TODO: Decide whether to store these separately or with the object they apply to.
type Event struct {
	TypeMeta `json:",inline"`

	// Required. The object that this event is about.
	InvolvedObject ObjectReference `json:"involvedObject,omitempty" description:"object that this event is about"`

	// Should be a short, machine understandable string that describes the current status
	// of the referred object. This should not give the reason for being in this state.
	// Examples: "Running", "CantStart", "CantSchedule", "Deleted".
	// It's OK for components to make up statuses to report here, but the same string should
	// always be used for the same status.
	// TODO: define a way of making sure these are consistent and don't collide.
	// TODO: provide exact specification for format.
	// DEPRECATED: Status (a.k.a Condition) value will be ignored.
	Status string `json:"status,omitempty" description:"short, machine understandable string that describes the current status of the referred object"`

	// Optional; this should be a short, machine understandable string that gives the reason
	// for the transition into the object's current status. For example, if ObjectStatus is
	// "CantStart", Reason might be "ImageNotFound".
	// TODO: provide exact specification for format.
	Reason string `json:"reason,omitempty" description:"short, machine understandable string that gives the reason for the transition into the object's current status"`

	// Optional. A human-readable description of the status of this operation.
	// TODO: decide on maximum length.
	Message string `json:"message,omitempty" description:"human-readable description of the status of this operation"`

	// Optional. The component reporting this event. Should be a short machine understandable string.
	// TODO: provide exact specification for format.
	Source string `json:"source,omitempty" description:"component reporting this event; short machine understandable string"`
	// Host name on which the event is generated.
	Host string `json:"host,omitempty" description:"host name on which this event was generated"`

	// The time at which the client recorded the event. (Time of server receipt is in TypeMeta.)
	Timestamp util.Time `json:"timestamp,omitempty" description:"time at which the client recorded the event"`
}

// EventList is a list of events.
type EventList struct {
	TypeMeta `json:",inline"`
	Items    []Event `json:"items" description:"list of events"`
}

// Backported from v1beta3 to replace ContainerManifest

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
	Volumes       []Volume      `json:"volumes" description:"list of volumes that can be mounted by containers belonging to the pod"`
	Containers    []Container   `json:"containers" description:"list of containers belonging to the pod"`
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty" description:"restart policy for all containers within the pod; one of RestartPolicyAlways, RestartPolicyOnFailure, RestartPolicyNever"`
	// Optional: Set DNS policy.  Defaults to "ClusterFirst"
	DNSPolicy DNSPolicy `json:"dnsPolicy,omitempty" description:"DNS policy for containers within the pod; one of 'ClusterFirst' or 'Default'"`
	// NodeSelector is a selector which must be true for the pod to fit on a node
	NodeSelector map[string]string `json:"nodeSelector,omitempty" description:"selector which must match a node's labels for the pod to be scheduled on that node"`

	// Host is a request to schedule this pod onto a specific host.  If it is non-empty,
	// the the scheduler simply schedules this pod onto that host, assuming that it fits
	// resource requirements.
	Host string `json:"host,omitempty" description:"host requested for this pod"`
}

// BoundPod is a collection of containers that should be run on a host. A BoundPod
// defines how a Pod may change after a Binding is created. A Pod is a request to
// execute a pod, whereas a BoundPod is the specification that would be run on a server.
type BoundPod struct {
	TypeMeta `json:",inline"`

	// Spec defines the behavior of a pod.
	Spec PodSpec `json:"spec,omitempty" description:"specification of the desired state of containers and volumes comprising the pod"`
}

// BoundPods is a list of Pods bound to a common server. The resource version of
// the pod list is guaranteed to only change when the list of bound pods changes.
type BoundPods struct {
	TypeMeta `json:",inline"`

	// Host is the name of a node that these pods were bound to.
	Host string `json:"host" description:"name of a node that these pods were bound to"`

	// Items is the list of all pods bound to a given host.
	Items []BoundPod `json:"items" description:"list of all pods bound to a given host"`
}

// List holds a list of objects, which may not be known by the server.
type List struct {
	TypeMeta `json:",inline"`
	Items    []runtime.RawExtension `json:"items" description:"list of objects"`
}
