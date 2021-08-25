/*
Copyright 2021 The Kubernetes Authors.

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

// This package contains intermediate types which will be mapped to their
// corresponding CRI `v1` or `v1alpha2` types. Base for this package are the v1
// definitions.
package cri

// This file contains all constants defined in CRI.

// Required runtime condition type.
const (
	// RuntimeReady means the runtime is up and ready to accept basic containers.
	RuntimeReady = "RuntimeReady"
	// NetworkReady means the runtime network is up and ready to accept containers which require network.
	NetworkReady = "NetworkReady"
)

// LogStreamType is the type of the stream in CRI container log.
type LogStreamType string

const (
	// Stdout is the stream type for stdout.
	Stdout LogStreamType = "stdout"
	// Stderr is the stream type for stderr.
	Stderr LogStreamType = "stderr"
)

// LogTag is the tag of a log line in CRI container log.
// Currently defined log tags:
// * First tag: Partial/Full - P/F.
// The field in the container log format can be extended to include multiple
// tags by using a delimiter, but changes should be rare. If it becomes clear
// that better extensibility is desired, a more extensible format (e.g., json)
// should be adopted as a replacement and/or addition.
type LogTag string

const (
	// LogTagPartial means the line is part of multiple lines.
	LogTagPartial LogTag = "P"
	// LogTagFull means the line is a single full line or the end of multiple lines.
	LogTagFull LogTag = "F"
	// LogTagDelimiter is the delimiter for different log tags.
	LogTagDelimiter = ":"
)

type Protocol int32

const (
	Protocol_TCP  Protocol = 0
	Protocol_UDP  Protocol = 1
	Protocol_SCTP Protocol = 2
)

var Protocol_name = map[int32]string{
	0: "TCP",
	1: "UDP",
	2: "SCTP",
}

var Protocol_value = map[string]int32{
	"TCP":  0,
	"UDP":  1,
	"SCTP": 2,
}

type MountPropagation int32

const (
	// No mount propagation ("private" in Linux terminology).
	MountPropagation_PROPAGATION_PRIVATE MountPropagation = 0
	// Mounts get propagated from the host to the container ("rslave" in Linux).
	MountPropagation_PROPAGATION_HOST_TO_CONTAINER MountPropagation = 1
	// Mounts get propagated from the host to the container and from the
	// container to the host ("rshared" in Linux).
	MountPropagation_PROPAGATION_BIDIRECTIONAL MountPropagation = 2
)

var MountPropagation_name = map[int32]string{
	0: "PROPAGATION_PRIVATE",
	1: "PROPAGATION_HOST_TO_CONTAINER",
	2: "PROPAGATION_BIDIRECTIONAL",
}

var MountPropagation_value = map[string]int32{
	"PROPAGATION_PRIVATE":           0,
	"PROPAGATION_HOST_TO_CONTAINER": 1,
	"PROPAGATION_BIDIRECTIONAL":     2,
}

// A NamespaceMode describes the intended namespace configuration for each
// of the namespaces (Network, PID, IPC) in NamespaceOption. Runtimes should
// map these modes as appropriate for the technology underlying the runtime.
type NamespaceMode int32

const (
	// A POD namespace is common to all containers in a pod.
	// For example, a container with a PID namespace of POD expects to view
	// all of the processes in all of the containers in the pod.
	NamespaceMode_POD NamespaceMode = 0
	// A CONTAINER namespace is restricted to a single container.
	// For example, a container with a PID namespace of CONTAINER expects to
	// view only the processes in that container.
	NamespaceMode_CONTAINER NamespaceMode = 1
	// A NODE namespace is the namespace of the Kubernetes node.
	// For example, a container with a PID namespace of NODE expects to view
	// all of the processes on the host running the kubelet.
	NamespaceMode_NODE NamespaceMode = 2
	// TARGET targets the namespace of another container. When this is specified,
	// a target_id must be specified in NamespaceOption and refer to a container
	// previously created with NamespaceMode CONTAINER. This containers namespace
	// will be made to match that of container target_id.
	// For example, a container with a PID namespace of TARGET expects to view
	// all of the processes that container target_id can view.
	NamespaceMode_TARGET NamespaceMode = 3
)

var NamespaceMode_name = map[int32]string{
	0: "POD",
	1: "CONTAINER",
	2: "NODE",
	3: "TARGET",
}

var NamespaceMode_value = map[string]int32{
	"POD":       0,
	"CONTAINER": 1,
	"NODE":      2,
	"TARGET":    3,
}

type PodSandboxState int32

const (
	PodSandboxState_SANDBOX_READY    PodSandboxState = 0
	PodSandboxState_SANDBOX_NOTREADY PodSandboxState = 1
)

var PodSandboxState_name = map[int32]string{
	0: "SANDBOX_READY",
	1: "SANDBOX_NOTREADY",
}

var PodSandboxState_value = map[string]int32{
	"SANDBOX_READY":    0,
	"SANDBOX_NOTREADY": 1,
}

type ContainerState int32

const (
	ContainerState_CONTAINER_CREATED ContainerState = 0
	ContainerState_CONTAINER_RUNNING ContainerState = 1
	ContainerState_CONTAINER_EXITED  ContainerState = 2
	ContainerState_CONTAINER_UNKNOWN ContainerState = 3
)

var ContainerState_name = map[int32]string{
	0: "CONTAINER_CREATED",
	1: "CONTAINER_RUNNING",
	2: "CONTAINER_EXITED",
	3: "CONTAINER_UNKNOWN",
}

var ContainerState_value = map[string]int32{
	"CONTAINER_CREATED": 0,
	"CONTAINER_RUNNING": 1,
	"CONTAINER_EXITED":  2,
	"CONTAINER_UNKNOWN": 3,
}

// Available profile types.
type SecurityProfile_ProfileType int32

const (
	// The container runtime default profile should be used.
	SecurityProfile_RuntimeDefault SecurityProfile_ProfileType = 0
	// Disable the feature for the sandbox or the container.
	SecurityProfile_Unconfined SecurityProfile_ProfileType = 1
	// A pre-defined profile on the node should be used.
	SecurityProfile_Localhost SecurityProfile_ProfileType = 2
)

var SecurityProfile_ProfileType_name = map[int32]string{
	0: "RuntimeDefault",
	1: "Unconfined",
	2: "Localhost",
}

var SecurityProfile_ProfileType_value = map[string]int32{
	"RuntimeDefault": 0,
	"Unconfined":     1,
	"Localhost":      2,
}

type VersionRequest struct {
	// Version of the kubelet runtime API.
	Version string `json:"version,omitempty"`
}

type VersionResponse struct {
	// Version of the kubelet runtime API.
	Version string `json:"version,omitempty"`
	// Name of the container runtime.
	RuntimeName string `json:"runtime_name,omitempty"`
	// Version of the container runtime. The string must be
	// semver-compatible.
	RuntimeVersion string `json:"runtime_version,omitempty"`
	// API version of the container runtime. The string must be
	// semver-compatible.
	RuntimeApiVersion string `json:"runtime_api_version,omitempty"`
}

// DNSConfig specifies the DNS servers and search domains of a sandbox.
type DNSConfig struct {
	// List of DNS servers of the cluster.
	Servers []string `json:"servers,omitempty"`
	// List of DNS search domains of the cluster.
	Searches []string `json:"searches,omitempty"`
	// List of DNS options. See https://linux.die.net/man/5/resolv.conf
	// for all available options.
	Options []string `json:"options,omitempty"`
}

// PortMapping specifies the port mapping configurations of a sandbox.
type PortMapping struct {
	// Protocol of the port mapping.
	Protocol Protocol `json:"protocol,omitempty"`
	// Port number within the container. Default: 0 (not specified).
	ContainerPort int32 `json:"container_port,omitempty"`
	// Port number on the host. Default: 0 (not specified).
	HostPort int32 `json:"host_port,omitempty"`
	// Host IP.
	HostIp string `json:"host_ip,omitempty"`
}

// Mount specifies a host volume to mount into a container.
type Mount struct {
	// Path of the mount within the container.
	ContainerPath string `json:"container_path,omitempty"`
	// Path of the mount on the host. If the hostPath doesn't exist, then runtimes
	// should report error. If the hostpath is a symbolic link, runtimes should
	// follow the symlink and mount the real destination to container.
	HostPath string `json:"host_path,omitempty"`
	// If set, the mount is read-only.
	Readonly bool `json:"readonly,omitempty"`
	// If set, the mount needs SELinux relabeling.
	SelinuxRelabel bool `json:"selinux_relabel,omitempty"`
	// Requested propagation mode.
	Propagation MountPropagation `json:"propagation,omitempty"`
}

// NamespaceOption provides options for Linux namespaces.
type NamespaceOption struct {
	// Network namespace for this container/sandbox.
	// Note: There is currently no way to set CONTAINER scoped network in the Kubernetes API.
	// Namespaces currently set by the kubelet: POD, NODE
	Network NamespaceMode `json:"network,omitempty"`
	// PID namespace for this container/sandbox.
	// Note: The CRI default is POD, but the v1.PodSpec default is CONTAINER.
	// The kubelet's runtime manager will set this to CONTAINER explicitly for v1 pods.
	// Namespaces currently set by the kubelet: POD, CONTAINER, NODE, TARGET
	Pid NamespaceMode `json:"pid,omitempty"`
	// IPC namespace for this container/sandbox.
	// Note: There is currently no way to set CONTAINER scoped IPC in the Kubernetes API.
	// Namespaces currently set by the kubelet: POD, NODE
	Ipc NamespaceMode `json:"ipc,omitempty"`
	// Target Container ID for NamespaceMode of TARGET. This container must have been
	// previously created in the same pod. It is not possible to specify different targets
	// for each namespace.
	TargetId string `json:"target_id,omitempty"`
}

// Int64Value is the wrapper of int64.
type Int64Value struct {
	// The value.
	Value int64 `json:"value,omitempty"`
}

// LinuxSandboxSecurityContext holds linux security configuration that will be
// applied to a sandbox. Note that:
// 1) It does not apply to containers in the pods.
// 2) It may not be applicable to a PodSandbox which does not contain any running
//    process.
type LinuxSandboxSecurityContext struct {
	// Configurations for the sandbox's namespaces.
	// This will be used only if the PodSandbox uses namespace for isolation.
	NamespaceOptions *NamespaceOption `json:"namespace_options,omitempty"`
	// Optional SELinux context to be applied.
	SelinuxOptions *SELinuxOption `json:"selinux_options,omitempty"`
	// UID to run sandbox processes as, when applicable.
	RunAsUser *Int64Value `json:"run_as_user,omitempty"`
	// GID to run sandbox processes as, when applicable. run_as_group should only
	// be specified when run_as_user is specified; otherwise, the runtime MUST error.
	RunAsGroup *Int64Value `json:"run_as_group,omitempty"`
	// If set, the root filesystem of the sandbox is read-only.
	ReadonlyRootfs bool `json:"readonly_rootfs,omitempty"`
	// List of groups applied to the first process run in the sandbox, in
	// addition to the sandbox's primary GID.
	SupplementalGroups []int64 `json:"supplemental_groups,omitempty"`
	// Indicates whether the sandbox will be asked to run a privileged
	// container. If a privileged container is to be executed within it, this
	// MUST be true.
	// This allows a sandbox to take additional security precautions if no
	// privileged containers are expected to be run.
	Privileged bool `json:"privileged,omitempty"`
	// Seccomp profile for the sandbox.
	Seccomp *SecurityProfile `json:"seccomp,omitempty"`
	// AppArmor profile for the sandbox.
	Apparmor *SecurityProfile `json:"apparmor,omitempty"`
	// Seccomp profile for the sandbox, candidate values are:
	// * runtime/default: the default profile for the container runtime
	// * unconfined: unconfined profile, ie, no seccomp sandboxing
	// * localhost/<full-path-to-profile>: the profile installed on the node.
	//   <full-path-to-profile> is the full path of the profile.
	// Default: "", which is identical with unconfined.
	SeccompProfilePath string `json:"seccomp_profile_path,omitempty"` // Deprecated: Do not use.
}

// A security profile which can be used for sandboxes and containers.
type SecurityProfile struct {
	// Indicator which `ProfileType` should be applied.
	ProfileType SecurityProfile_ProfileType `json:"profile_type,omitempty"`
	// Indicates that a pre-defined profile on the node should be used.
	// Must only be set if `ProfileType` is `Localhost`.
	// For seccomp, it must be an absolute path to the seccomp profile.
	// For AppArmor, this field is the AppArmor `<profile name>/`
	LocalhostRef string `json:"localhost_ref,omitempty"`
}

// LinuxPodSandboxConfig holds platform-specific configurations for Linux
// host platforms and Linux-based containers.
type LinuxPodSandboxConfig struct {
	// Parent cgroup of the PodSandbox.
	// The cgroupfs style syntax will be used, but the container runtime can
	// convert it to systemd semantics if needed.
	CgroupParent string `json:"cgroup_parent,omitempty"`
	// LinuxSandboxSecurityContext holds sandbox security attributes.
	SecurityContext *LinuxSandboxSecurityContext `json:"security_context,omitempty"`
	// Sysctls holds linux sysctls config for the sandbox.
	Sysctls map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Optional overhead represents the overheads associated with this sandbox
	Overhead *LinuxContainerResources `json:"overhead,omitempty"`
	// Optional resources represents the sum of container resources for this sandbox
	Resources *LinuxContainerResources `json:"resources,omitempty"`
}

// PodSandboxMetadata holds all necessary information for building the sandbox name.
// The container runtime is encouraged to expose the metadata associated with the
// PodSandbox in its user interface for better user experience. For example,
// the runtime can construct a unique PodSandboxName based on the metadata.
type PodSandboxMetadata struct {
	// Pod name of the sandbox. Same as the pod name in the Pod ObjectMeta.
	Name string `json:"name,omitempty"`
	// Pod UID of the sandbox. Same as the pod UID in the Pod ObjectMeta.
	Uid string `json:"uid,omitempty"`
	// Pod namespace of the sandbox. Same as the pod namespace in the Pod ObjectMeta.
	Namespace string `json:"namespace,omitempty"`
	// Attempt number of creating the sandbox. Default: 0.
	Attempt uint32 `json:"attempt,omitempty"`
}

// PodSandboxConfig holds all the required and optional fields for creating a
// sandbox.
type PodSandboxConfig struct {
	// Metadata of the sandbox. This information will uniquely identify the
	// sandbox, and the runtime should leverage this to ensure correct
	// operation. The runtime may also use this information to improve UX, such
	// as by constructing a readable name.
	Metadata *PodSandboxMetadata `json:"metadata,omitempty"`
	// Hostname of the sandbox. Hostname could only be empty when the pod
	// network namespace is NODE.
	Hostname string `json:"hostname,omitempty"`
	// Path to the directory on the host in which container log files are
	// stored.
	// By default the log of a container going into the LogDirectory will be
	// hooked up to STDOUT and STDERR. However, the LogDirectory may contain
	// binary log files with structured logging data from the individual
	// containers. For example, the files might be newline separated JSON
	// structured logs, systemd-journald journal files, gRPC trace files, etc.
	// E.g.,
	//     PodSandboxConfig.LogDirectory = `/var/log/pods/<podUID>/`
	//     ContainerConfig.LogPath = `containerName/Instance#.log`
	//
	// WARNING: Log management and how kubelet should interface with the
	// container logs are under active discussion in
	// https://issues.k8s.io/24677. There *may* be future change of direction
	// for logging as the discussion carries on.
	LogDirectory string `json:"log_directory,omitempty"`
	// DNS config for the sandbox.
	DnsConfig *DNSConfig `json:"dns_config,omitempty"`
	// Port mappings for the sandbox.
	PortMappings []*PortMapping `json:"port_mappings,omitempty"`
	// Key-value pairs that may be used to scope and select individual resources.
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map that may be set by the kubelet to store and
	// retrieve arbitrary metadata. This will include any annotations set on a
	// pod through the Kubernetes API.
	//
	// Annotations MUST NOT be altered by the runtime; the annotations stored
	// here MUST be returned in the PodSandboxStatus associated with the pod
	// this PodSandboxConfig creates.
	//
	// In general, in order to preserve a well-defined interface between the
	// kubelet and the container runtime, annotations SHOULD NOT influence
	// runtime behaviour.
	//
	// Annotations can also be useful for runtime authors to experiment with
	// new features that are opaque to the Kubernetes APIs (both user-facing
	// and the CRI). Whenever possible, however, runtime authors SHOULD
	// consider proposing new typed fields for any new features instead.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Optional configurations specific to Linux hosts.
	Linux *LinuxPodSandboxConfig `json:"linux,omitempty"`
	// Optional configurations specific to Windows hosts.
	Windows *WindowsPodSandboxConfig `json:"windows,omitempty"`
}

type RunPodSandboxRequest struct {
	// Configuration for creating a PodSandbox.
	Config *PodSandboxConfig `json:"config,omitempty"`
	// Named runtime configuration to use for this PodSandbox.
	// If the runtime handler is unknown, this request should be rejected.  An
	// empty string should select the default handler, equivalent to the
	// behavior before this feature was added.
	// See https://git.k8s.io/enhancements/keps/sig-node/585-runtime-class
	RuntimeHandler string `json:"runtime_handler,omitempty"`
}

type RunPodSandboxResponse struct {
	// ID of the PodSandbox to run.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
}

type StopPodSandboxRequest struct {
	// ID of the PodSandbox to stop.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
}

type StopPodSandboxResponse struct {
}

type RemovePodSandboxRequest struct {
	// ID of the PodSandbox to remove.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
}

type RemovePodSandboxResponse struct {
}

type PodSandboxStatusRequest struct {
	// ID of the PodSandbox for which to retrieve status.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
	// Verbose indicates whether to return extra information about the pod sandbox.
	Verbose bool `json:"verbose,omitempty"`
}

// PodIP represents an ip of a Pod
type PodIP struct {
	// an ip is a string representation of an IPv4 or an IPv6
	Ip string `json:"ip,omitempty"`
}

// PodSandboxNetworkStatus is the status of the network for a PodSandbox.
type PodSandboxNetworkStatus struct {
	// IP address of the PodSandbox.
	Ip string `json:"ip,omitempty"`
	// list of additional ips (not inclusive of PodSandboxNetworkStatus.Ip) of the PodSandBoxNetworkStatus
	AdditionalIps []*PodIP `json:"additional_ips,omitempty"`
}

// Namespace contains paths to the namespaces.
type Namespace struct {
	// Namespace options for Linux namespaces.
	Options *NamespaceOption `json:"options,omitempty"`
}

// LinuxSandboxStatus contains status specific to Linux sandboxes.
type LinuxPodSandboxStatus struct {
	// Paths to the sandbox's namespaces.
	Namespaces *Namespace `json:"namespaces,omitempty"`
}

// PodSandboxStatus contains the status of the PodSandbox.
type PodSandboxStatus struct {
	// ID of the sandbox.
	Id string `json:"id,omitempty"`
	// Metadata of the sandbox.
	Metadata *PodSandboxMetadata `json:"metadata,omitempty"`
	// State of the sandbox.
	State PodSandboxState `json:"state,omitempty"`
	// Creation timestamp of the sandbox in nanoseconds. Must be > 0.
	CreatedAt int64 `json:"created_at,omitempty"`
	// Network contains network status if network is handled by the runtime.
	Network *PodSandboxNetworkStatus `json:"network,omitempty"`
	// Linux-specific status to a pod sandbox.
	Linux *LinuxPodSandboxStatus `json:"linux,omitempty"`
	// Labels are key-value pairs that may be used to scope and select individual resources.
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map holding arbitrary metadata.
	// Annotations MUST NOT be altered by the runtime; the value of this field
	// MUST be identical to that of the corresponding PodSandboxConfig used to
	// instantiate the pod sandbox this status represents.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// runtime configuration used for this PodSandbox.
	RuntimeHandler string `json:"runtime_handler,omitempty"`
}

type PodSandboxStatusResponse struct {
	// Status of the PodSandbox.
	Status *PodSandboxStatus `json:"status,omitempty"`
	// Info is extra information of the PodSandbox. The key could be arbitrary string, and
	// value should be in json format. The information could include anything useful for
	// debug, e.g. network namespace for linux container based container runtime.
	// It should only be returned non-empty when Verbose is true.
	Info map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

// PodSandboxStateValue is the wrapper of PodSandboxState.
type PodSandboxStateValue struct {
	// State of the sandbox.
	State PodSandboxState `json:"state,omitempty"`
}

// PodSandboxFilter is used to filter a list of PodSandboxes.
// All those fields are combined with 'AND'
type PodSandboxFilter struct {
	// ID of the sandbox.
	Id string `json:"id,omitempty"`
	// State of the sandbox.
	State *PodSandboxStateValue `json:"state,omitempty"`
	// LabelSelector to select matches.
	// Only api.MatchLabels is supported for now and the requirements
	// are ANDed. MatchExpressions is not supported yet.
	LabelSelector map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type ListPodSandboxRequest struct {
	// PodSandboxFilter to filter a list of PodSandboxes.
	Filter *PodSandboxFilter `json:"filter,omitempty"`
}

// PodSandbox contains minimal information about a sandbox.
type PodSandbox struct {
	// ID of the PodSandbox.
	Id string `json:"id,omitempty"`
	// Metadata of the PodSandbox.
	Metadata *PodSandboxMetadata `json:"metadata,omitempty"`
	// State of the PodSandbox.
	State PodSandboxState `json:"state,omitempty"`
	// Creation timestamps of the PodSandbox in nanoseconds. Must be > 0.
	CreatedAt int64 `json:"created_at,omitempty"`
	// Labels of the PodSandbox.
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map holding arbitrary metadata.
	// Annotations MUST NOT be altered by the runtime; the value of this field
	// MUST be identical to that of the corresponding PodSandboxConfig used to
	// instantiate this PodSandbox.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// runtime configuration used for this PodSandbox.
	RuntimeHandler string `json:"runtime_handler,omitempty"`
}

type ListPodSandboxResponse struct {
	// List of PodSandboxes.
	Items []*PodSandbox `json:"items,omitempty"`
}

type PodSandboxStatsRequest struct {
	// ID of the pod sandbox for which to retrieve stats.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
}

type PodSandboxStatsResponse struct {
	Stats *PodSandboxStats `json:"stats,omitempty"`
}

// PodSandboxStatsFilter is used to filter pod sandboxes.
// All those fields are combined with 'AND'.
type PodSandboxStatsFilter struct {
	// ID of the pod sandbox.
	Id string `json:"id,omitempty"`
	// LabelSelector to select matches.
	// Only api.MatchLabels is supported for now and the requirements
	// are ANDed. MatchExpressions is not supported yet.
	LabelSelector map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type ListPodSandboxStatsRequest struct {
	// Filter for the list request.
	Filter *PodSandboxStatsFilter `json:"filter,omitempty"`
}

type ListPodSandboxStatsResponse struct {
	// Stats of the pod sandbox.
	Stats []*PodSandboxStats `json:"stats,omitempty"`
}

// PodSandboxAttributes provides basic information of the pod sandbox.
type PodSandboxAttributes struct {
	// ID of the pod sandbox.
	Id string `json:"id,omitempty"`
	// Metadata of the pod sandbox.
	Metadata *PodSandboxMetadata `json:"metadata,omitempty"`
	// Key-value pairs that may be used to scope and select individual resources.
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map holding arbitrary metadata.
	// Annotations MUST NOT be altered by the runtime; the value of this field
	// MUST be identical to that of the corresponding PodSandboxStatus used to
	// instantiate the PodSandbox this status represents.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

// PodSandboxStats provides the resource usage statistics for a pod.
// The linux or windows field will be populated depending on the platform.
type PodSandboxStats struct {
	// Information of the pod.
	Attributes *PodSandboxAttributes `json:"attributes,omitempty"`
	// Stats from linux.
	Linux *LinuxPodSandboxStats `json:"linux,omitempty"`
	// Stats from windows.
	Windows *WindowsPodSandboxStats `json:"windows,omitempty"`
}

// LinuxPodSandboxStats provides the resource usage statistics for a pod sandbox on linux.
type LinuxPodSandboxStats struct {
	// CPU usage gathered for the pod sandbox.
	Cpu *CpuUsage `json:"cpu,omitempty"`
	// Memory usage gathered for the pod sandbox.
	Memory *MemoryUsage `json:"memory,omitempty"`
	// Network usage gathered for the pod sandbox
	Network *NetworkUsage `json:"network,omitempty"`
	// Stats pertaining to processes in the pod sandbox.
	Process *ProcessUsage `json:"process,omitempty"`
	// Stats of containers in the measured pod sandbox.
	Containers []*ContainerStats `json:"containers,omitempty"`
}

// WindowsPodSandboxStats provides the resource usage statistics for a pod sandbox on windows
type WindowsPodSandboxStats struct {
}

// NetworkUsage contains data about network resources.
type NetworkUsage struct {
	// The time at which these stats were updated.
	Timestamp int64 `json:"timestamp,omitempty"`
	// Stats for the default network interface.
	DefaultInterface *NetworkInterfaceUsage `json:"default_interface,omitempty"`
	// Stats for all found network interfaces, excluding the default.
	Interfaces []*NetworkInterfaceUsage `json:"interfaces,omitempty"`
}

// NetworkInterfaceUsage contains resource value data about a network interface.
type NetworkInterfaceUsage struct {
	// The name of the network interface.
	Name string `json:"name,omitempty"`
	// Cumulative count of bytes received.
	RxBytes *UInt64Value `json:"rx_bytes,omitempty"`
	// Cumulative count of receive errors encountered.
	RxErrors *UInt64Value `json:"rx_errors,omitempty"`
	// Cumulative count of bytes transmitted.
	TxBytes *UInt64Value `json:"tx_bytes,omitempty"`
	// Cumulative count of transmit errors encountered.
	TxErrors *UInt64Value `json:"tx_errors,omitempty"`
}

// ProcessUsage are stats pertaining to processes.
type ProcessUsage struct {
	// The time at which these stats were updated.
	Timestamp int64 `json:"timestamp,omitempty"`
	// Number of processes.
	ProcessCount *UInt64Value `json:"process_count,omitempty"`
}

// ImageSpec is an internal representation of an image.
type ImageSpec struct {
	// Container's Image field (e.g. imageID or imageDigest).
	Image string `json:"image,omitempty"`
	// Unstructured key-value map holding arbitrary metadata.
	// ImageSpec Annotations can be used to help the runtime target specific
	// images in multi-arch images.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type KeyValue struct {
	Key   string `json:"key,omitempty"`
	Value string `json:"value,omitempty"`
}

// LinuxContainerResources specifies Linux specific configuration for
// resources.
// TODO: Consider using Resources from opencontainers/runtime-spec/specs-go
// directly.
type LinuxContainerResources struct {
	// CPU CFS (Completely Fair Scheduler) period. Default: 0 (not specified).
	CpuPeriod int64 `json:"cpu_period,omitempty"`
	// CPU CFS (Completely Fair Scheduler) quota. Default: 0 (not specified).
	CpuQuota int64 `json:"cpu_quota,omitempty"`
	// CPU shares (relative weight vs. other containers). Default: 0 (not specified).
	CpuShares int64 `json:"cpu_shares,omitempty"`
	// Memory limit in bytes. Default: 0 (not specified).
	MemoryLimitInBytes int64 `json:"memory_limit_in_bytes,omitempty"`
	// OOMScoreAdj adjusts the oom-killer score. Default: 0 (not specified).
	OomScoreAdj int64 `json:"oom_score_adj,omitempty"`
	// CpusetCpus constrains the allowed set of logical CPUs. Default: "" (not specified).
	CpusetCpus string `json:"cpuset_cpus,omitempty"`
	// CpusetMems constrains the allowed set of memory nodes. Default: "" (not specified).
	CpusetMems string `json:"cpuset_mems,omitempty"`
	// List of HugepageLimits to limit the HugeTLB usage of container per page size. Default: nil (not specified).
	HugepageLimits []*HugepageLimit `json:"hugepage_limits,omitempty"`
	// Unified resources for cgroup v2. Default: nil (not specified).
	// Each key/value in the map refers to the cgroup v2.
	// e.g. "memory.max": "6937202688" or "io.weight": "default 100".
	Unified map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Memory swap limit in bytes. Default 0 (not specified).
	MemorySwapLimitInBytes int64 `json:"memory_swap_limit_in_bytes,omitempty"`
}

// HugepageLimit corresponds to the file`hugetlb.<hugepagesize>.limit_in_byte` in container level cgroup.
// For example, `PageSize=1GB`, `Limit=1073741824` means setting `1073741824` bytes to hugetlb.1GB.limit_in_bytes.
type HugepageLimit struct {
	// The value of PageSize has the format <size><unit-prefix>B (2MB, 1GB),
	// and must match the <hugepagesize> of the corresponding control file found in `hugetlb.<hugepagesize>.limit_in_bytes`.
	// The values of <unit-prefix> are intended to be parsed using base 1024("1KB" = 1024, "1MB" = 1048576, etc).
	PageSize string `json:"page_size,omitempty"`
	// limit in bytes of hugepagesize HugeTLB usage.
	Limit uint64 `json:"limit,omitempty"`
}

// SELinuxOption are the labels to be applied to the container.
type SELinuxOption struct {
	User  string `json:"user,omitempty"`
	Role  string `json:"role,omitempty"`
	Type  string `json:"type,omitempty"`
	Level string `json:"level,omitempty"`
}

// Capability contains the container capabilities to add or drop
type Capability struct {
	// List of capabilities to add.
	AddCapabilities []string `json:"add_capabilities,omitempty"`
	// List of capabilities to drop.
	DropCapabilities []string `json:"drop_capabilities,omitempty"`
}

// LinuxContainerSecurityContext holds linux security configuration that will be applied to a container.
type LinuxContainerSecurityContext struct {
	// Capabilities to add or drop.
	Capabilities *Capability `json:"capabilities,omitempty"`
	// If set, run container in privileged mode.
	// Privileged mode is incompatible with the following options. If
	// privileged is set, the following features MAY have no effect:
	// 1. capabilities
	// 2. selinux_options
	// 4. seccomp
	// 5. apparmor
	//
	// Privileged mode implies the following specific options are applied:
	// 1. All capabilities are added.
	// 2. Sensitive paths, such as kernel module paths within sysfs, are not masked.
	// 3. Any sysfs and procfs mounts are mounted RW.
	// 4. AppArmor confinement is not applied.
	// 5. Seccomp restrictions are not applied.
	// 6. The device cgroup does not restrict access to any devices.
	// 7. All devices from the host's /dev are available within the container.
	// 8. SELinux restrictions are not applied (e.g. label=disabled).
	Privileged bool `json:"privileged,omitempty"`
	// Configurations for the container's namespaces.
	// Only used if the container uses namespace for isolation.
	NamespaceOptions *NamespaceOption `json:"namespace_options,omitempty"`
	// SELinux context to be optionally applied.
	SelinuxOptions *SELinuxOption `json:"selinux_options,omitempty"`
	// UID to run the container process as. Only one of run_as_user and
	// run_as_username can be specified at a time.
	RunAsUser *Int64Value `json:"run_as_user,omitempty"`
	// GID to run the container process as. run_as_group should only be specified
	// when run_as_user or run_as_username is specified; otherwise, the runtime
	// MUST error.
	RunAsGroup *Int64Value `json:"run_as_group,omitempty"`
	// User name to run the container process as. If specified, the user MUST
	// exist in the container image (i.e. in the /etc/passwd inside the image),
	// and be resolved there by the runtime; otherwise, the runtime MUST error.
	RunAsUsername string `json:"run_as_username,omitempty"`
	// If set, the root filesystem of the container is read-only.
	ReadonlyRootfs bool `json:"readonly_rootfs,omitempty"`
	// List of groups applied to the first process run in the container, in
	// addition to the container's primary GID.
	SupplementalGroups []int64 `json:"supplemental_groups,omitempty"`
	// no_new_privs defines if the flag for no_new_privs should be set on the
	// container.
	NoNewPrivs bool `json:"no_new_privs,omitempty"`
	// masked_paths is a slice of paths that should be masked by the container
	// runtime, this can be passed directly to the OCI spec.
	MaskedPaths []string `json:"masked_paths,omitempty"`
	// readonly_paths is a slice of paths that should be set as readonly by the
	// container runtime, this can be passed directly to the OCI spec.
	ReadonlyPaths []string `json:"readonly_paths,omitempty"`
	// Seccomp profile for the container.
	Seccomp *SecurityProfile `json:"seccomp,omitempty"`
	// AppArmor profile for the container.
	Apparmor *SecurityProfile `json:"apparmor,omitempty"`
	// AppArmor profile for the container, candidate values are:
	// * runtime/default: equivalent to not specifying a profile.
	// * unconfined: no profiles are loaded
	// * localhost/<profile_name>: profile loaded on the node
	//    (localhost) by name. The possible profile names are detailed at
	//    https://gitlab.com/apparmor/apparmor/-/wikis/AppArmor_Core_Policy_Reference
	ApparmorProfile string `json:"apparmor_profile,omitempty"` // Deprecated: Do not use.
	// Seccomp profile for the container, candidate values are:
	// * runtime/default: the default profile for the container runtime
	// * unconfined: unconfined profile, ie, no seccomp sandboxing
	// * localhost/<full-path-to-profile>: the profile installed on the node.
	//   <full-path-to-profile> is the full path of the profile.
	// Default: "", which is identical with unconfined.
	SeccompProfilePath string `json:"seccomp_profile_path,omitempty"` // Deprecated: Do not use.
}

// LinuxContainerConfig contains platform-specific configuration for
// Linux-based containers.
type LinuxContainerConfig struct {
	// Resources specification for the container.
	Resources *LinuxContainerResources `json:"resources,omitempty"`
	// LinuxContainerSecurityContext configuration for the container.
	SecurityContext *LinuxContainerSecurityContext `json:"security_context,omitempty"`
}

// WindowsSandboxSecurityContext holds platform-specific configurations that will be
// applied to a sandbox.
// These settings will only apply to the sandbox container.
type WindowsSandboxSecurityContext struct {
	// User name to run the container process as. If specified, the user MUST
	// exist in the container image and be resolved there by the runtime;
	// otherwise, the runtime MUST return error.
	RunAsUsername string `json:"run_as_username,omitempty"`
	// The contents of the GMSA credential spec to use to run this container.
	CredentialSpec string `json:"credential_spec,omitempty"`
	// Indicates whether the container be asked to run as a HostProcess container.
	HostProcess bool `json:"host_process,omitempty"`
}

// WindowsPodSandboxConfig holds platform-specific configurations for Windows
// host platforms and Windows-based containers.
type WindowsPodSandboxConfig struct {
	// WindowsSandboxSecurityContext holds sandbox security attributes.
	SecurityContext *WindowsSandboxSecurityContext `json:"security_context,omitempty"`
}

// WindowsContainerSecurityContext holds windows security configuration that will be applied to a container.
type WindowsContainerSecurityContext struct {
	// User name to run the container process as. If specified, the user MUST
	// exist in the container image and be resolved there by the runtime;
	// otherwise, the runtime MUST return error.
	RunAsUsername string `json:"run_as_username,omitempty"`
	// The contents of the GMSA credential spec to use to run this container.
	CredentialSpec string `json:"credential_spec,omitempty"`
	// Indicates whether a container is to be run as a HostProcess container.
	HostProcess bool `json:"host_process,omitempty"`
}

// WindowsContainerConfig contains platform-specific configuration for
// Windows-based containers.
type WindowsContainerConfig struct {
	// Resources specification for the container.
	Resources *WindowsContainerResources `json:"resources,omitempty"`
	// WindowsContainerSecurityContext configuration for the container.
	SecurityContext *WindowsContainerSecurityContext `json:"security_context,omitempty"`
}

// WindowsContainerResources specifies Windows specific configuration for
// resources.
type WindowsContainerResources struct {
	// CPU shares (relative weight vs. other containers). Default: 0 (not specified).
	CpuShares int64 `json:"cpu_shares,omitempty"`
	// Number of CPUs available to the container. Default: 0 (not specified).
	CpuCount int64 `json:"cpu_count,omitempty"`
	// Specifies the portion of processor cycles that this container can use as a percentage times 100.
	CpuMaximum int64 `json:"cpu_maximum,omitempty"`
	// Memory limit in bytes. Default: 0 (not specified).
	MemoryLimitInBytes int64 `json:"memory_limit_in_bytes,omitempty"`
}

// ContainerMetadata holds all necessary information for building the container
// name. The container runtime is encouraged to expose the metadata in its user
// interface for better user experience. E.g., runtime can construct a unique
// container name based on the metadata. Note that (name, attempt) is unique
// within a sandbox for the entire lifetime of the sandbox.
type ContainerMetadata struct {
	// Name of the container. Same as the container name in the PodSpec.
	Name string `json:"name,omitempty"`
	// Attempt number of creating the container. Default: 0.
	Attempt uint32 `json:"attempt,omitempty"`
}

// Device specifies a host device to mount into a container.
type Device struct {
	// Path of the device within the container.
	ContainerPath string `json:"container_path,omitempty"`
	// Path of the device on the host.
	HostPath string `json:"host_path,omitempty"`
	// Cgroups permissions of the device, candidates are one or more of
	// * r - allows container to read from the specified device.
	// * w - allows container to write to the specified device.
	// * m - allows container to create device files that do not yet exist.
	Permissions string `json:"permissions,omitempty"`
}

// ContainerConfig holds all the required and optional fields for creating a
// container.
type ContainerConfig struct {
	// Metadata of the container. This information will uniquely identify the
	// container, and the runtime should leverage this to ensure correct
	// operation. The runtime may also use this information to improve UX, such
	// as by constructing a readable name.
	Metadata *ContainerMetadata `json:"metadata,omitempty"`
	// Image to use.
	Image *ImageSpec `json:"image,omitempty"`
	// Command to execute (i.e., entrypoint for docker)
	Command []string `json:"command,omitempty"`
	// Args for the Command (i.e., command for docker)
	Args []string `json:"args,omitempty"`
	// Current working directory of the command.
	WorkingDir string `json:"working_dir,omitempty"`
	// List of environment variable to set in the container.
	Envs []*KeyValue `json:"envs,omitempty"`
	// Mounts for the container.
	Mounts []*Mount `json:"mounts,omitempty"`
	// Devices for the container.
	Devices []*Device `json:"devices,omitempty"`
	// Key-value pairs that may be used to scope and select individual resources.
	// Label keys are of the form:
	//     label-key ::= prefixed-name | name
	//     prefixed-name ::= prefix '/' name
	//     prefix ::= DNS_SUBDOMAIN
	//     name ::= DNS_LABEL
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map that may be used by the kubelet to store and
	// retrieve arbitrary metadata.
	//
	// Annotations MUST NOT be altered by the runtime; the annotations stored
	// here MUST be returned in the ContainerStatus associated with the container
	// this ContainerConfig creates.
	//
	// In general, in order to preserve a well-defined interface between the
	// kubelet and the container runtime, annotations SHOULD NOT influence
	// runtime behaviour.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Path relative to PodSandboxConfig.LogDirectory for container to store
	// the log (STDOUT and STDERR) on the host.
	// E.g.,
	//     PodSandboxConfig.LogDirectory = `/var/log/pods/<podUID>/`
	//     ContainerConfig.LogPath = `containerName/Instance#.log`
	//
	// WARNING: Log management and how kubelet should interface with the
	// container logs are under active discussion in
	// https://issues.k8s.io/24677. There *may* be future change of direction
	// for logging as the discussion carries on.
	LogPath string `json:"log_path,omitempty"`
	// Variables for interactive containers, these have very specialized
	// use-cases (e.g. debugging).
	// TODO: Determine if we need to continue supporting these fields that are
	// part of Kubernetes's Container Spec.
	Stdin     bool `json:"stdin,omitempty"`
	StdinOnce bool `json:"stdin_once,omitempty"`
	Tty       bool `json:"tty,omitempty"`
	// Configuration specific to Linux containers.
	Linux *LinuxContainerConfig `json:"linux,omitempty"`
	// Configuration specific to Windows containers.
	Windows *WindowsContainerConfig `json:"windows,omitempty"`
}

type CreateContainerRequest struct {
	// ID of the PodSandbox in which the container should be created.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
	// Config of the container.
	Config *ContainerConfig `json:"config,omitempty"`
	// Config of the PodSandbox. This is the same config that was passed
	// to RunPodSandboxRequest to create the PodSandbox. It is passed again
	// here just for easy reference. The PodSandboxConfig is immutable and
	// remains the same throughout the lifetime of the pod.
	SandboxConfig *PodSandboxConfig `json:"sandbox_config,omitempty"`
}

type CreateContainerResponse struct {
	// ID of the created container.
	ContainerId string `json:"container_id,omitempty"`
}

type StartContainerRequest struct {
	// ID of the container to start.
	ContainerId string `json:"container_id,omitempty"`
}

type StartContainerResponse struct {
}

type StopContainerRequest struct {
	// ID of the container to stop.
	ContainerId string `json:"container_id,omitempty"`
	// Timeout in seconds to wait for the container to stop before forcibly
	// terminating it. Default: 0 (forcibly terminate the container immediately)
	Timeout int64 `json:"timeout,omitempty"`
}

type StopContainerResponse struct {
}

type RemoveContainerRequest struct {
	// ID of the container to remove.
	ContainerId string `json:"container_id,omitempty"`
}

type RemoveContainerResponse struct {
}

// ContainerStateValue is the wrapper of ContainerState.
type ContainerStateValue struct {
	// State of the container.
	State ContainerState `json:"state,omitempty"`
}

// ContainerFilter is used to filter containers.
// All those fields are combined with 'AND'
type ContainerFilter struct {
	// ID of the container.
	Id string `json:"id,omitempty"`
	// State of the container.
	State *ContainerStateValue `json:"state,omitempty"`
	// ID of the PodSandbox.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
	// LabelSelector to select matches.
	// Only api.MatchLabels is supported for now and the requirements
	// are ANDed. MatchExpressions is not supported yet.
	LabelSelector map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type ListContainersRequest struct {
	Filter *ContainerFilter `json:"filter,omitempty"`
}

// Container provides the runtime information for a container, such as ID, hash,
// state of the container.
type Container struct {
	// ID of the container, used by the container runtime to identify
	// a container.
	Id string `json:"id,omitempty"`
	// ID of the sandbox to which this container belongs.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
	// Metadata of the container.
	Metadata *ContainerMetadata `json:"metadata,omitempty"`
	// Spec of the image.
	Image *ImageSpec `json:"image,omitempty"`
	// Reference to the image in use. For most runtimes, this should be an
	// image ID.
	ImageRef string `json:"image_ref,omitempty"`
	// State of the container.
	State ContainerState `json:"state,omitempty"`
	// Creation time of the container in nanoseconds.
	CreatedAt int64 `json:"created_at,omitempty"`
	// Key-value pairs that may be used to scope and select individual resources.
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map holding arbitrary metadata.
	// Annotations MUST NOT be altered by the runtime; the value of this field
	// MUST be identical to that of the corresponding ContainerConfig used to
	// instantiate this Container.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type ListContainersResponse struct {
	// List of containers.
	Containers []*Container `json:"containers,omitempty"`
}

type ContainerStatusRequest struct {
	// ID of the container for which to retrieve status.
	ContainerId string `json:"container_id,omitempty"`
	// Verbose indicates whether to return extra information about the container.
	Verbose bool `json:"verbose,omitempty"`
}

// ContainerStatus represents the status of a container.
type ContainerStatus struct {
	// ID of the container.
	Id string `json:"id,omitempty"`
	// Metadata of the container.
	Metadata *ContainerMetadata `json:"metadata,omitempty"`
	// Status of the container.
	State ContainerState `json:"state,omitempty"`
	// Creation time of the container in nanoseconds.
	CreatedAt int64 `json:"created_at,omitempty"`
	// Start time of the container in nanoseconds. Default: 0 (not specified).
	StartedAt int64 `json:"started_at,omitempty"`
	// Finish time of the container in nanoseconds. Default: 0 (not specified).
	FinishedAt int64 `json:"finished_at,omitempty"`
	// Exit code of the container. Only required when finished_at != 0. Default: 0.
	ExitCode int32 `json:"exit_code,omitempty"`
	// Spec of the image.
	Image *ImageSpec `json:"image,omitempty"`
	// Reference to the image in use. For most runtimes, this should be an
	// image ID
	ImageRef string `json:"image_ref,omitempty"`
	// Brief CamelCase string explaining why container is in its current state.
	Reason string `json:"reason,omitempty"`
	// Human-readable message indicating details about why container is in its
	// current state.
	Message string `json:"message,omitempty"`
	// Key-value pairs that may be used to scope and select individual resources.
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map holding arbitrary metadata.
	// Annotations MUST NOT be altered by the runtime; the value of this field
	// MUST be identical to that of the corresponding ContainerConfig used to
	// instantiate the Container this status represents.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Mounts for the container.
	Mounts []*Mount `json:"mounts,omitempty"`
	// Log path of container.
	LogPath string `json:"log_path,omitempty"`
}

type ContainerStatusResponse struct {
	// Status of the container.
	Status *ContainerStatus `json:"status,omitempty"`
	// Info is extra information of the Container. The key could be arbitrary string, and
	// value should be in json format. The information could include anything useful for
	// debug, e.g. pid for linux container based container runtime.
	// It should only be returned non-empty when Verbose is true.
	Info map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type UpdateContainerResourcesRequest struct {
	// ID of the container to update.
	ContainerId string `json:"container_id,omitempty"`
	// Resource configuration specific to Linux containers.
	Linux *LinuxContainerResources `json:"linux,omitempty"`
	// Resource configuration specific to Windows containers.
	Windows *WindowsContainerResources `json:"windows,omitempty"`
	// Unstructured key-value map holding arbitrary additional information for
	// container resources updating. This can be used for specifying experimental
	// resources to update or other options to use when updating the container.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type UpdateContainerResourcesResponse struct {
}

type ExecSyncRequest struct {
	// ID of the container.
	ContainerId string `json:"container_id,omitempty"`
	// Command to execute.
	Cmd []string `json:"cmd,omitempty"`
	// Timeout in seconds to stop the command. Default: 0 (run forever).
	Timeout int64 `json:"timeout,omitempty"`
}

type ExecSyncResponse struct {
	// Captured command stdout output.
	Stdout []byte `json:"stdout,omitempty"`
	// Captured command stderr output.
	Stderr []byte `json:"stderr,omitempty"`
	// Exit code the command finished with. Default: 0 (success).
	ExitCode int32 `json:"exit_code,omitempty"`
}

type ExecRequest struct {
	// ID of the container in which to execute the command.
	ContainerId string `json:"container_id,omitempty"`
	// Command to execute.
	Cmd []string `json:"cmd,omitempty"`
	// Whether to exec the command in a TTY.
	Tty bool `json:"tty,omitempty"`
	// Whether to stream stdin.
	// One of `stdin`, `stdout`, and `stderr` MUST be true.
	Stdin bool `json:"stdin,omitempty"`
	// Whether to stream stdout.
	// One of `stdin`, `stdout`, and `stderr` MUST be true.
	Stdout bool `json:"stdout,omitempty"`
	// Whether to stream stderr.
	// One of `stdin`, `stdout`, and `stderr` MUST be true.
	// If `tty` is true, `stderr` MUST be false. Multiplexing is not supported
	// in this case. The output of stdout and stderr will be combined to a
	// single stream.
	Stderr bool `json:"stderr,omitempty"`
}

type ExecResponse struct {
	// Fully qualified URL of the exec streaming server.
	Url string `json:"url,omitempty"`
}

type AttachRequest struct {
	// ID of the container to which to attach.
	ContainerId string `json:"container_id,omitempty"`
	// Whether to stream stdin.
	// One of `stdin`, `stdout`, and `stderr` MUST be true.
	Stdin bool `json:"stdin,omitempty"`
	// Whether the process being attached is running in a TTY.
	// This must match the TTY setting in the ContainerConfig.
	Tty bool `json:"tty,omitempty"`
	// Whether to stream stdout.
	// One of `stdin`, `stdout`, and `stderr` MUST be true.
	Stdout bool `json:"stdout,omitempty"`
	// Whether to stream stderr.
	// One of `stdin`, `stdout`, and `stderr` MUST be true.
	// If `tty` is true, `stderr` MUST be false. Multiplexing is not supported
	// in this case. The output of stdout and stderr will be combined to a
	// single stream.
	Stderr bool `json:"stderr,omitempty"`
}

type AttachResponse struct {
	// Fully qualified URL of the attach streaming server.
	Url string `json:"url,omitempty"`
}

type PortForwardRequest struct {
	// ID of the container to which to forward the port.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
	// Port to forward.
	Port []int32 `json:"port,omitempty"`
}

type PortForwardResponse struct {
	// Fully qualified URL of the port-forward streaming server.
	Url string `json:"url,omitempty"`
}

type ImageFilter struct {
	// Spec of the image.
	Image *ImageSpec `json:"image,omitempty"`
}

type ListImagesRequest struct {
	// Filter to list images.
	Filter *ImageFilter `json:"filter,omitempty"`
}

// Basic information about a container image.
type Image struct {
	// ID of the image.
	Id string `json:"id,omitempty"`
	// Other names by which this image is known.
	RepoTags []string `json:"repo_tags,omitempty"`
	// Digests by which this image is known.
	RepoDigests []string `json:"repo_digests,omitempty"`
	// Size of the image in bytes. Must be > 0.
	Size_ uint64 `json:"size,omitempty"`
	// UID that will run the command(s). This is used as a default if no user is
	// specified when creating the container. UID and the following user name
	// are mutually exclusive.
	Uid *Int64Value `json:"uid,omitempty"`
	// User name that will run the command(s). This is used if UID is not set
	// and no user is specified when creating container.
	Username string `json:"username,omitempty"`
	// ImageSpec for image which includes annotations
	Spec *ImageSpec `json:"spec,omitempty"`
}

type ListImagesResponse struct {
	// List of images.
	Images []*Image `json:"images,omitempty"`
}

type ImageStatusRequest struct {
	// Spec of the image.
	Image *ImageSpec `json:"image,omitempty"`
	// Verbose indicates whether to return extra information about the image.
	Verbose bool `json:"verbose,omitempty"`
}

type ImageStatusResponse struct {
	// Status of the image.
	Image *Image `json:"image,omitempty"`
	// Info is extra information of the Image. The key could be arbitrary string, and
	// value should be in json format. The information could include anything useful
	// for debug, e.g. image config for oci image based container runtime.
	// It should only be returned non-empty when Verbose is true.
	Info map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

// AuthConfig contains authorization information for connecting to a registry.
type AuthConfig struct {
	Username      string `json:"username,omitempty"`
	Password      string `json:"password,omitempty"`
	Auth          string `json:"auth,omitempty"`
	ServerAddress string `json:"server_address,omitempty"`
	// IdentityToken is used to authenticate the user and get
	// an access token for the registry.
	IdentityToken string `json:"identity_token,omitempty"`
	// RegistryToken is a bearer token to be sent to a registry
	RegistryToken string `json:"registry_token,omitempty"`
}

type PullImageRequest struct {
	// Spec of the image.
	Image *ImageSpec `json:"image,omitempty"`
	// Authentication configuration for pulling the image.
	Auth *AuthConfig `json:"auth,omitempty"`
	// Config of the PodSandbox, which is used to pull image in PodSandbox context.
	SandboxConfig *PodSandboxConfig `json:"sandbox_config,omitempty"`
}

type PullImageResponse struct {
	// Reference to the image in use. For most runtimes, this should be an
	// image ID or digest.
	ImageRef string `json:"image_ref,omitempty"`
}

type RemoveImageRequest struct {
	// Spec of the image to remove.
	Image *ImageSpec `json:"image,omitempty"`
}

type RemoveImageResponse struct {
}

type NetworkConfig struct {
	// CIDR to use for pod IP addresses. If the CIDR is empty, runtimes
	// should omit it.
	PodCidr string `json:"pod_cidr,omitempty"`
}

type RuntimeConfig struct {
	NetworkConfig *NetworkConfig `json:"network_config,omitempty"`
}

type UpdateRuntimeConfigRequest struct {
	RuntimeConfig *RuntimeConfig `json:"runtime_config,omitempty"`
}

type UpdateRuntimeConfigResponse struct {
}

// RuntimeCondition contains condition information for the runtime.
// There are 2 kinds of runtime conditions:
// 1. Required conditions: Conditions are required for kubelet to work
// properly. If any required condition is unmet, the node will be not ready.
// The required conditions include:
//   * RuntimeReady: RuntimeReady means the runtime is up and ready to accept
//   basic containers e.g. container only needs host network.
//   * NetworkReady: NetworkReady means the runtime network is up and ready to
//   accept containers which require container network.
// 2. Optional conditions: Conditions are informative to the user, but kubelet
// will not rely on. Since condition type is an arbitrary string, all conditions
// not required are optional. These conditions will be exposed to users to help
// them understand the status of the system.
type RuntimeCondition struct {
	// Type of runtime condition.
	Type string `json:"type,omitempty"`
	// Status of the condition, one of true/false. Default: false.
	Status bool `json:"status,omitempty"`
	// Brief CamelCase string containing reason for the condition's last transition.
	Reason string `json:"reason,omitempty"`
	// Human-readable message indicating details about last transition.
	Message string `json:"message,omitempty"`
}

// RuntimeStatus is information about the current status of the runtime.
type RuntimeStatus struct {
	// List of current observed runtime conditions.
	Conditions []*RuntimeCondition `json:"conditions,omitempty"`
}

type StatusRequest struct {
	// Verbose indicates whether to return extra information about the runtime.
	Verbose bool `json:"verbose,omitempty"`
}

type StatusResponse struct {
	// Status of the Runtime.
	Status *RuntimeStatus `json:"status,omitempty"`
	// Info is extra information of the Runtime. The key could be arbitrary string, and
	// value should be in json format. The information could include anything useful for
	// debug, e.g. plugins used by the container runtime.
	// It should only be returned non-empty when Verbose is true.
	Info map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type ImageFsInfoRequest struct {
}

// UInt64Value is the wrapper of uint64.
type UInt64Value struct {
	// The value.
	Value uint64 `json:"value,omitempty"`
}

// FilesystemIdentifier uniquely identify the filesystem.
type FilesystemIdentifier struct {
	// Mountpoint of a filesystem.
	Mountpoint string `json:"mountpoint,omitempty"`
}

// FilesystemUsage provides the filesystem usage information.
type FilesystemUsage struct {
	// Timestamp in nanoseconds at which the information were collected. Must be > 0.
	Timestamp int64 `json:"timestamp,omitempty"`
	// The unique identifier of the filesystem.
	FsId *FilesystemIdentifier `json:"fs_id,omitempty"`
	// UsedBytes represents the bytes used for images on the filesystem.
	// This may differ from the total bytes used on the filesystem and may not
	// equal CapacityBytes - AvailableBytes.
	UsedBytes *UInt64Value `json:"used_bytes,omitempty"`
	// InodesUsed represents the inodes used by the images.
	// This may not equal InodesCapacity - InodesAvailable because the underlying
	// filesystem may also be used for purposes other than storing images.
	InodesUsed *UInt64Value `json:"inodes_used,omitempty"`
}

type ImageFsInfoResponse struct {
	// Information of image filesystem(s).
	ImageFilesystems []*FilesystemUsage `json:"image_filesystems,omitempty"`
}

type ContainerStatsRequest struct {
	// ID of the container for which to retrieve stats.
	ContainerId string `json:"container_id,omitempty"`
}

type ContainerStatsResponse struct {
	// Stats of the container.
	Stats *ContainerStats `json:"stats,omitempty"`
}

type ListContainerStatsRequest struct {
	// Filter for the list request.
	Filter *ContainerStatsFilter `json:"filter,omitempty"`
}

// ContainerStatsFilter is used to filter containers.
// All those fields are combined with 'AND'
type ContainerStatsFilter struct {
	// ID of the container.
	Id string `json:"id,omitempty"`
	// ID of the PodSandbox.
	PodSandboxId string `json:"pod_sandbox_id,omitempty"`
	// LabelSelector to select matches.
	// Only api.MatchLabels is supported for now and the requirements
	// are ANDed. MatchExpressions is not supported yet.
	LabelSelector map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

type ListContainerStatsResponse struct {
	// Stats of the container.
	Stats []*ContainerStats `json:"stats,omitempty"`
}

// ContainerAttributes provides basic information of the container.
type ContainerAttributes struct {
	// ID of the container.
	Id string `json:"id,omitempty"`
	// Metadata of the container.
	Metadata *ContainerMetadata `json:"metadata,omitempty"`
	// Key-value pairs that may be used to scope and select individual resources.
	Labels map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
	// Unstructured key-value map holding arbitrary metadata.
	// Annotations MUST NOT be altered by the runtime; the value of this field
	// MUST be identical to that of the corresponding ContainerConfig used to
	// instantiate the Container this status represents.
	Annotations map[string]string `protobuf_val:"bytes,2,opt,name=value,proto3"`
}

// ContainerStats provides the resource usage statistics for a container.
type ContainerStats struct {
	// Information of the container.
	Attributes *ContainerAttributes `json:"attributes,omitempty"`
	// CPU usage gathered from the container.
	Cpu *CpuUsage `json:"cpu,omitempty"`
	// Memory usage gathered from the container.
	Memory *MemoryUsage `json:"memory,omitempty"`
	// Usage of the writable layer.
	WritableLayer *FilesystemUsage `json:"writable_layer,omitempty"`
}

// CpuUsage provides the CPU usage information.
type CpuUsage struct {
	// Timestamp in nanoseconds at which the information were collected. Must be > 0.
	Timestamp int64 `json:"timestamp,omitempty"`
	// Cumulative CPU usage (sum across all cores) since object creation.
	UsageCoreNanoSeconds *UInt64Value `json:"usage_core_nano_seconds,omitempty"`
	// Total CPU usage (sum of all cores) averaged over the sample window.
	// The "core" unit can be interpreted as CPU core-nanoseconds per second.
	UsageNanoCores *UInt64Value `json:"usage_nano_cores,omitempty"`
}

// MemoryUsage provides the memory usage information.
type MemoryUsage struct {
	// Timestamp in nanoseconds at which the information were collected. Must be > 0.
	Timestamp int64 `json:"timestamp,omitempty"`
	// The amount of working set memory in bytes.
	WorkingSetBytes *UInt64Value `json:"working_set_bytes,omitempty"`
	// Available memory for use.  This is defined as the memory limit = workingSetBytes.
	AvailableBytes *UInt64Value `json:"available_bytes,omitempty"`
	// Total memory in use. This includes all memory regardless of when it was accessed.
	UsageBytes *UInt64Value `json:"usage_bytes,omitempty"`
	// The amount of anonymous and swap cache memory (includes transparent hugepages).
	RssBytes *UInt64Value `json:"rss_bytes,omitempty"`
	// Cumulative number of minor page faults.
	PageFaults *UInt64Value `json:"page_faults,omitempty"`
	// Cumulative number of major page faults.
	MajorPageFaults *UInt64Value `json:"major_page_faults,omitempty"`
}

type ReopenContainerLogRequest struct {
	// ID of the container for which to reopen the log.
	ContainerId string `json:"container_id,omitempty"`
}

type ReopenContainerLogResponse struct{}

func (m *VersionRequest) GetVersion() string {
	if m != nil {
		return m.Version
	}
	return ""
}

func (m *VersionResponse) GetVersion() string {
	if m != nil {
		return m.Version
	}
	return ""
}

func (m *VersionResponse) GetRuntimeName() string {
	if m != nil {
		return m.RuntimeName
	}
	return ""
}

func (m *VersionResponse) GetRuntimeVersion() string {
	if m != nil {
		return m.RuntimeVersion
	}
	return ""
}

func (m *VersionResponse) GetRuntimeApiVersion() string {
	if m != nil {
		return m.RuntimeApiVersion
	}
	return ""
}

func (m *DNSConfig) GetServers() []string {
	if m != nil {
		return m.Servers
	}
	return nil
}

func (m *DNSConfig) GetSearches() []string {
	if m != nil {
		return m.Searches
	}
	return nil
}

func (m *DNSConfig) GetOptions() []string {
	if m != nil {
		return m.Options
	}
	return nil
}

func (m *PortMapping) GetProtocol() Protocol {
	if m != nil {
		return m.Protocol
	}
	return Protocol_TCP
}

func (m *PortMapping) GetContainerPort() int32 {
	if m != nil {
		return m.ContainerPort
	}
	return 0
}

func (m *PortMapping) GetHostPort() int32 {
	if m != nil {
		return m.HostPort
	}
	return 0
}

func (m *PortMapping) GetHostIp() string {
	if m != nil {
		return m.HostIp
	}
	return ""
}

func (m *Mount) GetContainerPath() string {
	if m != nil {
		return m.ContainerPath
	}
	return ""
}

func (m *Mount) GetHostPath() string {
	if m != nil {
		return m.HostPath
	}
	return ""
}

func (m *Mount) GetReadonly() bool {
	if m != nil {
		return m.Readonly
	}
	return false
}

func (m *Mount) GetSelinuxRelabel() bool {
	if m != nil {
		return m.SelinuxRelabel
	}
	return false
}

func (m *Mount) GetPropagation() MountPropagation {
	if m != nil {
		return m.Propagation
	}
	return MountPropagation_PROPAGATION_PRIVATE
}

func (m *Namespace) GetOptions() *NamespaceOption {
	if m != nil {
		return m.Options
	}
	return nil
}

func (m *NamespaceOption) GetNetwork() NamespaceMode {
	if m != nil {
		return m.Network
	}
	return NamespaceMode_POD
}

func (m *NamespaceOption) GetPid() NamespaceMode {
	if m != nil {
		return m.Pid
	}
	return NamespaceMode_POD
}

func (m *NamespaceOption) GetIpc() NamespaceMode {
	if m != nil {
		return m.Ipc
	}
	return NamespaceMode_POD
}

func (m *NamespaceOption) GetTargetId() string {
	if m != nil {
		return m.TargetId
	}
	return ""
}

func (m *Int64Value) GetValue() int64 {
	if m != nil {
		return m.Value
	}
	return 0
}

func (m *LinuxSandboxSecurityContext) GetNamespaceOptions() *NamespaceOption {
	if m != nil {
		return m.NamespaceOptions
	}
	return nil
}

func (m *LinuxSandboxSecurityContext) GetSelinuxOptions() *SELinuxOption {
	if m != nil {
		return m.SelinuxOptions
	}
	return nil
}

func (m *LinuxSandboxSecurityContext) GetRunAsUser() *Int64Value {
	if m != nil {
		return m.RunAsUser
	}
	return nil
}

func (m *LinuxSandboxSecurityContext) GetRunAsGroup() *Int64Value {
	if m != nil {
		return m.RunAsGroup
	}
	return nil
}

func (m *LinuxSandboxSecurityContext) GetReadonlyRootfs() bool {
	if m != nil {
		return m.ReadonlyRootfs
	}
	return false
}

func (m *LinuxSandboxSecurityContext) GetSupplementalGroups() []int64 {
	if m != nil {
		return m.SupplementalGroups
	}
	return nil
}

func (m *LinuxSandboxSecurityContext) GetPrivileged() bool {
	if m != nil {
		return m.Privileged
	}
	return false
}

func (m *LinuxSandboxSecurityContext) GetSeccomp() *SecurityProfile {
	if m != nil {
		return m.Seccomp
	}
	return nil
}

func (m *LinuxSandboxSecurityContext) GetApparmor() *SecurityProfile {
	if m != nil {
		return m.Apparmor
	}
	return nil
}

// Deprecated: Do not use.
func (m *LinuxSandboxSecurityContext) GetSeccompProfilePath() string {
	if m != nil {
		return m.SeccompProfilePath
	}
	return ""
}

func (m *SecurityProfile) GetProfileType() SecurityProfile_ProfileType {
	if m != nil {
		return m.ProfileType
	}
	return SecurityProfile_RuntimeDefault
}

func (m *SecurityProfile) GetLocalhostRef() string {
	if m != nil {
		return m.LocalhostRef
	}
	return ""
}

func (m *LinuxPodSandboxConfig) GetCgroupParent() string {
	if m != nil {
		return m.CgroupParent
	}
	return ""
}

func (m *LinuxPodSandboxConfig) GetSecurityContext() *LinuxSandboxSecurityContext {
	if m != nil {
		return m.SecurityContext
	}
	return nil
}

func (m *LinuxPodSandboxConfig) GetSysctls() map[string]string {
	if m != nil {
		return m.Sysctls
	}
	return nil
}

func (m *LinuxPodSandboxConfig) GetOverhead() *LinuxContainerResources {
	if m != nil {
		return m.Overhead
	}
	return nil
}

func (m *LinuxPodSandboxConfig) GetResources() *LinuxContainerResources {
	if m != nil {
		return m.Resources
	}
	return nil
}

func (m *PodSandboxMetadata) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *PodSandboxMetadata) GetUid() string {
	if m != nil {
		return m.Uid
	}
	return ""
}

func (m *PodSandboxMetadata) GetNamespace() string {
	if m != nil {
		return m.Namespace
	}
	return ""
}

func (m *PodSandboxMetadata) GetAttempt() uint32 {
	if m != nil {
		return m.Attempt
	}
	return 0
}

func (m *PodSandboxConfig) GetMetadata() *PodSandboxMetadata {
	if m != nil {
		return m.Metadata
	}
	return nil
}

func (m *PodSandboxConfig) GetHostname() string {
	if m != nil {
		return m.Hostname
	}
	return ""
}

func (m *PodSandboxConfig) GetLogDirectory() string {
	if m != nil {
		return m.LogDirectory
	}
	return ""
}

func (m *PodSandboxConfig) GetDnsConfig() *DNSConfig {
	if m != nil {
		return m.DnsConfig
	}
	return nil
}

func (m *PodSandboxConfig) GetPortMappings() []*PortMapping {
	if m != nil {
		return m.PortMappings
	}
	return nil
}

func (m *PodSandboxConfig) GetLabels() map[string]string {
	if m != nil {
		return m.Labels
	}
	return nil
}

func (m *PodSandboxConfig) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *PodSandboxConfig) GetLinux() *LinuxPodSandboxConfig {
	if m != nil {
		return m.Linux
	}
	return nil
}

func (m *PodSandboxConfig) GetWindows() *WindowsPodSandboxConfig {
	if m != nil {
		return m.Windows
	}
	return nil
}

func (m *RunPodSandboxRequest) GetConfig() *PodSandboxConfig {
	if m != nil {
		return m.Config
	}
	return nil
}

func (m *RunPodSandboxRequest) GetRuntimeHandler() string {
	if m != nil {
		return m.RuntimeHandler
	}
	return ""
}

func (m *RunPodSandboxResponse) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *StopPodSandboxRequest) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *RemovePodSandboxRequest) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *PodSandboxStatusRequest) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *PodSandboxStatusRequest) GetVerbose() bool {
	if m != nil {
		return m.Verbose
	}
	return false
}

func (m *PodIP) GetIp() string {
	if m != nil {
		return m.Ip
	}
	return ""
}

func (m *PodSandboxNetworkStatus) GetIp() string {
	if m != nil {
		return m.Ip
	}
	return ""
}

func (m *PodSandboxNetworkStatus) GetAdditionalIps() []*PodIP {
	if m != nil {
		return m.AdditionalIps
	}
	return nil
}

func (m *LinuxPodSandboxStatus) GetNamespaces() *Namespace {
	if m != nil {
		return m.Namespaces
	}
	return nil
}

func (m *PodSandboxStatus) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *PodSandboxStatus) GetMetadata() *PodSandboxMetadata {
	if m != nil {
		return m.Metadata
	}
	return nil
}

func (m *PodSandboxStatus) GetState() PodSandboxState {
	if m != nil {
		return m.State
	}
	return PodSandboxState_SANDBOX_READY
}

func (m *PodSandboxStatus) GetCreatedAt() int64 {
	if m != nil {
		return m.CreatedAt
	}
	return 0
}

func (m *PodSandboxStatus) GetNetwork() *PodSandboxNetworkStatus {
	if m != nil {
		return m.Network
	}
	return nil
}

func (m *PodSandboxStatus) GetLinux() *LinuxPodSandboxStatus {
	if m != nil {
		return m.Linux
	}
	return nil
}

func (m *PodSandboxStatus) GetLabels() map[string]string {
	if m != nil {
		return m.Labels
	}
	return nil
}

func (m *PodSandboxStatus) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *PodSandboxStatus) GetRuntimeHandler() string {
	if m != nil {
		return m.RuntimeHandler
	}
	return ""
}

func (m *PodSandboxStatusResponse) GetStatus() *PodSandboxStatus {
	if m != nil {
		return m.Status
	}
	return nil
}

func (m *PodSandboxStatusResponse) GetInfo() map[string]string {
	if m != nil {
		return m.Info
	}
	return nil
}

func (m *PodSandboxFilter) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *PodSandboxFilter) GetState() *PodSandboxStateValue {
	if m != nil {
		return m.State
	}
	return nil
}

func (m *PodSandboxFilter) GetLabelSelector() map[string]string {
	if m != nil {
		return m.LabelSelector
	}
	return nil
}

func (m *ListPodSandboxRequest) GetFilter() *PodSandboxFilter {
	if m != nil {
		return m.Filter
	}
	return nil
}

func (m *PodSandbox) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *PodSandbox) GetMetadata() *PodSandboxMetadata {
	if m != nil {
		return m.Metadata
	}
	return nil
}

func (m *PodSandbox) GetState() PodSandboxState {
	if m != nil {
		return m.State
	}
	return PodSandboxState_SANDBOX_READY
}

func (m *PodSandbox) GetCreatedAt() int64 {
	if m != nil {
		return m.CreatedAt
	}
	return 0
}

func (m *PodSandbox) GetLabels() map[string]string {
	if m != nil {
		return m.Labels
	}
	return nil
}

func (m *PodSandbox) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *PodSandbox) GetRuntimeHandler() string {
	if m != nil {
		return m.RuntimeHandler
	}
	return ""
}

func (m *PodSandboxStatsRequest) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *PodSandboxStatsResponse) GetStats() *PodSandboxStats {
	if m != nil {
		return m.Stats
	}
	return nil
}

func (m *PodSandboxStatsFilter) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *PodSandboxStatsFilter) GetLabelSelector() map[string]string {
	if m != nil {
		return m.LabelSelector
	}
	return nil
}

func (m *ListPodSandboxStatsRequest) GetFilter() *PodSandboxStatsFilter {
	if m != nil {
		return m.Filter
	}
	return nil
}

func (m *ListPodSandboxStatsResponse) GetStats() []*PodSandboxStats {
	if m != nil {
		return m.Stats
	}
	return nil
}

func (m *PodSandboxStats) GetAttributes() *PodSandboxAttributes {
	if m != nil {
		return m.Attributes
	}
	return nil
}

func (m *PodSandboxStats) GetLinux() *LinuxPodSandboxStats {
	if m != nil {
		return m.Linux
	}
	return nil
}

func (m *PodSandboxStats) GetWindows() *WindowsPodSandboxStats {
	if m != nil {
		return m.Windows
	}
	return nil
}

func (m *LinuxPodSandboxStats) GetCpu() *CpuUsage {
	if m != nil {
		return m.Cpu
	}
	return nil
}

func (m *LinuxPodSandboxStats) GetMemory() *MemoryUsage {
	if m != nil {
		return m.Memory
	}
	return nil
}

func (m *LinuxPodSandboxStats) GetNetwork() *NetworkUsage {
	if m != nil {
		return m.Network
	}
	return nil
}

func (m *LinuxPodSandboxStats) GetProcess() *ProcessUsage {
	if m != nil {
		return m.Process
	}
	return nil
}

func (m *LinuxPodSandboxStats) GetContainers() []*ContainerStats {
	if m != nil {
		return m.Containers
	}
	return nil
}

func (m *NetworkUsage) GetTimestamp() int64 {
	if m != nil {
		return m.Timestamp
	}
	return 0
}

func (m *NetworkUsage) GetDefaultInterface() *NetworkInterfaceUsage {
	if m != nil {
		return m.DefaultInterface
	}
	return nil
}

func (m *NetworkUsage) GetInterfaces() []*NetworkInterfaceUsage {
	if m != nil {
		return m.Interfaces
	}
	return nil
}

func (m *NetworkInterfaceUsage) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *NetworkInterfaceUsage) GetRxBytes() *UInt64Value {
	if m != nil {
		return m.RxBytes
	}
	return nil
}

func (m *NetworkInterfaceUsage) GetRxErrors() *UInt64Value {
	if m != nil {
		return m.RxErrors
	}
	return nil
}

func (m *NetworkInterfaceUsage) GetTxBytes() *UInt64Value {
	if m != nil {
		return m.TxBytes
	}
	return nil
}

func (m *NetworkInterfaceUsage) GetTxErrors() *UInt64Value {
	if m != nil {
		return m.TxErrors
	}
	return nil
}

func (m *ProcessUsage) GetTimestamp() int64 {
	if m != nil {
		return m.Timestamp
	}
	return 0
}

func (m *ProcessUsage) GetProcessCount() *UInt64Value {
	if m != nil {
		return m.ProcessCount
	}
	return nil
}

func (m *ImageSpec) GetImage() string {
	if m != nil {
		return m.Image
	}
	return ""
}

func (m *ImageSpec) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *KeyValue) GetKey() string {
	if m != nil {
		return m.Key
	}
	return ""
}

func (m *KeyValue) GetValue() string {
	if m != nil {
		return m.Value
	}
	return ""
}

func (m *LinuxContainerResources) GetCpuPeriod() int64 {
	if m != nil {
		return m.CpuPeriod
	}
	return 0
}

func (m *LinuxContainerResources) GetCpuQuota() int64 {
	if m != nil {
		return m.CpuQuota
	}
	return 0
}

func (m *LinuxContainerResources) GetCpuShares() int64 {
	if m != nil {
		return m.CpuShares
	}
	return 0
}

func (m *LinuxContainerResources) GetMemoryLimitInBytes() int64 {
	if m != nil {
		return m.MemoryLimitInBytes
	}
	return 0
}

func (m *LinuxContainerResources) GetOomScoreAdj() int64 {
	if m != nil {
		return m.OomScoreAdj
	}
	return 0
}

func (m *LinuxContainerResources) GetCpusetCpus() string {
	if m != nil {
		return m.CpusetCpus
	}
	return ""
}

func (m *LinuxContainerResources) GetCpusetMems() string {
	if m != nil {
		return m.CpusetMems
	}
	return ""
}

func (m *LinuxContainerResources) GetHugepageLimits() []*HugepageLimit {
	if m != nil {
		return m.HugepageLimits
	}
	return nil
}

func (m *LinuxContainerResources) GetUnified() map[string]string {
	if m != nil {
		return m.Unified
	}
	return nil
}

func (m *LinuxContainerResources) GetMemorySwapLimitInBytes() int64 {
	if m != nil {
		return m.MemorySwapLimitInBytes
	}
	return 0
}

func (m *HugepageLimit) GetPageSize() string {
	if m != nil {
		return m.PageSize
	}
	return ""
}

func (m *HugepageLimit) GetLimit() uint64 {
	if m != nil {
		return m.Limit
	}
	return 0
}

func (m *SELinuxOption) GetUser() string {
	if m != nil {
		return m.User
	}
	return ""
}

func (m *SELinuxOption) GetRole() string {
	if m != nil {
		return m.Role
	}
	return ""
}

func (m *SELinuxOption) GetType() string {
	if m != nil {
		return m.Type
	}
	return ""
}

func (m *SELinuxOption) GetLevel() string {
	if m != nil {
		return m.Level
	}
	return ""
}

func (m *Capability) GetAddCapabilities() []string {
	if m != nil {
		return m.AddCapabilities
	}
	return nil
}

func (m *Capability) GetDropCapabilities() []string {
	if m != nil {
		return m.DropCapabilities
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetCapabilities() *Capability {
	if m != nil {
		return m.Capabilities
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetPrivileged() bool {
	if m != nil {
		return m.Privileged
	}
	return false
}

func (m *LinuxContainerSecurityContext) GetNamespaceOptions() *NamespaceOption {
	if m != nil {
		return m.NamespaceOptions
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetSelinuxOptions() *SELinuxOption {
	if m != nil {
		return m.SelinuxOptions
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetRunAsUser() *Int64Value {
	if m != nil {
		return m.RunAsUser
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetRunAsGroup() *Int64Value {
	if m != nil {
		return m.RunAsGroup
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetRunAsUsername() string {
	if m != nil {
		return m.RunAsUsername
	}
	return ""
}

func (m *LinuxContainerSecurityContext) GetReadonlyRootfs() bool {
	if m != nil {
		return m.ReadonlyRootfs
	}
	return false
}

func (m *LinuxContainerSecurityContext) GetSupplementalGroups() []int64 {
	if m != nil {
		return m.SupplementalGroups
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetNoNewPrivs() bool {
	if m != nil {
		return m.NoNewPrivs
	}
	return false
}

func (m *LinuxContainerSecurityContext) GetMaskedPaths() []string {
	if m != nil {
		return m.MaskedPaths
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetReadonlyPaths() []string {
	if m != nil {
		return m.ReadonlyPaths
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetSeccomp() *SecurityProfile {
	if m != nil {
		return m.Seccomp
	}
	return nil
}

func (m *LinuxContainerSecurityContext) GetApparmor() *SecurityProfile {
	if m != nil {
		return m.Apparmor
	}
	return nil
}

// Deprecated: Do not use.
func (m *LinuxContainerSecurityContext) GetApparmorProfile() string {
	if m != nil {
		return m.ApparmorProfile
	}
	return ""
}

// Deprecated: Do not use.
func (m *LinuxContainerSecurityContext) GetSeccompProfilePath() string {
	if m != nil {
		return m.SeccompProfilePath
	}
	return ""
}

func (m *LinuxContainerConfig) GetResources() *LinuxContainerResources {
	if m != nil {
		return m.Resources
	}
	return nil
}

func (m *LinuxContainerConfig) GetSecurityContext() *LinuxContainerSecurityContext {
	if m != nil {
		return m.SecurityContext
	}
	return nil
}

func (m *WindowsSandboxSecurityContext) GetRunAsUsername() string {
	if m != nil {
		return m.RunAsUsername
	}
	return ""
}

func (m *WindowsSandboxSecurityContext) GetCredentialSpec() string {
	if m != nil {
		return m.CredentialSpec
	}
	return ""
}

func (m *WindowsSandboxSecurityContext) GetHostProcess() bool {
	if m != nil {
		return m.HostProcess
	}
	return false
}

func (m *WindowsPodSandboxConfig) GetSecurityContext() *WindowsSandboxSecurityContext {
	if m != nil {
		return m.SecurityContext
	}
	return nil
}

func (m *WindowsContainerSecurityContext) GetRunAsUsername() string {
	if m != nil {
		return m.RunAsUsername
	}
	return ""
}

func (m *WindowsContainerSecurityContext) GetCredentialSpec() string {
	if m != nil {
		return m.CredentialSpec
	}
	return ""
}

func (m *WindowsContainerSecurityContext) GetHostProcess() bool {
	if m != nil {
		return m.HostProcess
	}
	return false
}

func (m *WindowsContainerConfig) GetResources() *WindowsContainerResources {
	if m != nil {
		return m.Resources
	}
	return nil
}

func (m *WindowsContainerConfig) GetSecurityContext() *WindowsContainerSecurityContext {
	if m != nil {
		return m.SecurityContext
	}
	return nil
}

func (m *WindowsContainerResources) GetCpuShares() int64 {
	if m != nil {
		return m.CpuShares
	}
	return 0
}

func (m *WindowsContainerResources) GetCpuCount() int64 {
	if m != nil {
		return m.CpuCount
	}
	return 0
}

func (m *WindowsContainerResources) GetCpuMaximum() int64 {
	if m != nil {
		return m.CpuMaximum
	}
	return 0
}

func (m *WindowsContainerResources) GetMemoryLimitInBytes() int64 {
	if m != nil {
		return m.MemoryLimitInBytes
	}
	return 0
}

func (m *ContainerMetadata) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *ContainerMetadata) GetAttempt() uint32 {
	if m != nil {
		return m.Attempt
	}
	return 0
}

func (m *Device) GetContainerPath() string {
	if m != nil {
		return m.ContainerPath
	}
	return ""
}

func (m *Device) GetHostPath() string {
	if m != nil {
		return m.HostPath
	}
	return ""
}

func (m *Device) GetPermissions() string {
	if m != nil {
		return m.Permissions
	}
	return ""
}

func (m *ContainerConfig) GetMetadata() *ContainerMetadata {
	if m != nil {
		return m.Metadata
	}
	return nil
}

func (m *ContainerConfig) GetImage() *ImageSpec {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *ContainerConfig) GetCommand() []string {
	if m != nil {
		return m.Command
	}
	return nil
}

func (m *ContainerConfig) GetArgs() []string {
	if m != nil {
		return m.Args
	}
	return nil
}

func (m *ContainerConfig) GetWorkingDir() string {
	if m != nil {
		return m.WorkingDir
	}
	return ""
}

func (m *ContainerConfig) GetEnvs() []*KeyValue {
	if m != nil {
		return m.Envs
	}
	return nil
}

func (m *ContainerConfig) GetMounts() []*Mount {
	if m != nil {
		return m.Mounts
	}
	return nil
}

func (m *ContainerConfig) GetDevices() []*Device {
	if m != nil {
		return m.Devices
	}
	return nil
}

func (m *ContainerConfig) GetLabels() map[string]string {
	if m != nil {
		return m.Labels
	}
	return nil
}

func (m *ContainerConfig) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *ContainerConfig) GetLogPath() string {
	if m != nil {
		return m.LogPath
	}
	return ""
}

func (m *ContainerConfig) GetStdin() bool {
	if m != nil {
		return m.Stdin
	}
	return false
}

func (m *ContainerConfig) GetStdinOnce() bool {
	if m != nil {
		return m.StdinOnce
	}
	return false
}

func (m *ContainerConfig) GetTty() bool {
	if m != nil {
		return m.Tty
	}
	return false
}

func (m *ContainerConfig) GetLinux() *LinuxContainerConfig {
	if m != nil {
		return m.Linux
	}
	return nil
}

func (m *ContainerConfig) GetWindows() *WindowsContainerConfig {
	if m != nil {
		return m.Windows
	}
	return nil
}

func (m *CreateContainerRequest) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *CreateContainerRequest) GetConfig() *ContainerConfig {
	if m != nil {
		return m.Config
	}
	return nil
}

func (m *CreateContainerRequest) GetSandboxConfig() *PodSandboxConfig {
	if m != nil {
		return m.SandboxConfig
	}
	return nil
}

func (m *StartContainerRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}

func (m *StopContainerRequest) GetTimeout() int64 {
	if m != nil {
		return m.Timeout
	}
	return 0
}

func (m *ContainerFilter) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *ContainerFilter) GetState() *ContainerStateValue {
	if m != nil {
		return m.State
	}
	return nil
}

func (m *ContainerFilter) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *ContainerFilter) GetLabelSelector() map[string]string {
	if m != nil {
		return m.LabelSelector
	}
	return nil
}

func (m *ListContainersRequest) GetFilter() *ContainerFilter {
	if m != nil {
		return m.Filter
	}
	return nil
}

func (m *Container) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *Container) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *Container) GetMetadata() *ContainerMetadata {
	if m != nil {
		return m.Metadata
	}
	return nil
}

func (m *Container) GetImage() *ImageSpec {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *Container) GetImageRef() string {
	if m != nil {
		return m.ImageRef
	}
	return ""
}

func (m *Container) GetState() ContainerState {
	if m != nil {
		return m.State
	}
	return ContainerState_CONTAINER_CREATED
}

func (m *Container) GetCreatedAt() int64 {
	if m != nil {
		return m.CreatedAt
	}
	return 0
}

func (m *Container) GetLabels() map[string]string {
	if m != nil {
		return m.Labels
	}
	return nil
}

func (m *Container) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *ListContainersResponse) GetContainers() []*Container {
	if m != nil {
		return m.Containers
	}
	return nil
}

func (m *ContainerStatusRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}

func (m *ContainerStatus) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *ContainerStatus) GetMetadata() *ContainerMetadata {
	if m != nil {
		return m.Metadata
	}
	return nil
}

func (m *ContainerStatus) GetState() ContainerState {
	if m != nil {
		return m.State
	}
	return ContainerState_CONTAINER_CREATED
}

func (m *ContainerStatus) GetCreatedAt() int64 {
	if m != nil {
		return m.CreatedAt
	}
	return 0
}

func (m *ContainerStatus) GetStartedAt() int64 {
	if m != nil {
		return m.StartedAt
	}
	return 0
}

func (m *ContainerStatus) GetFinishedAt() int64 {
	if m != nil {
		return m.FinishedAt
	}
	return 0
}

func (m *ContainerStatus) GetExitCode() int32 {
	if m != nil {
		return m.ExitCode
	}
	return 0
}

func (m *ContainerStatus) GetImage() *ImageSpec {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *ContainerStatus) GetImageRef() string {
	if m != nil {
		return m.ImageRef
	}
	return ""
}

func (m *ContainerStatus) GetReason() string {
	if m != nil {
		return m.Reason
	}
	return ""
}

func (m *ContainerStatus) GetMessage() string {
	if m != nil {
		return m.Message
	}
	return ""
}

func (m *ContainerStatus) GetLabels() map[string]string {
	if m != nil {
		return m.Labels
	}
	return nil
}

func (m *ContainerStatus) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *ContainerStatus) GetMounts() []*Mount {
	if m != nil {
		return m.Mounts
	}
	return nil
}

func (m *ContainerStatus) GetLogPath() string {
	if m != nil {
		return m.LogPath
	}
	return ""
}

func (m *ContainerStatusResponse) GetStatus() *ContainerStatus {
	if m != nil {
		return m.Status
	}
	return nil
}

func (m *ContainerStatusResponse) GetInfo() map[string]string {
	if m != nil {
		return m.Info
	}
	return nil
}

func (m *UpdateContainerResourcesRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}

func (m *UpdateContainerResourcesRequest) GetLinux() *LinuxContainerResources {
	if m != nil {
		return m.Linux
	}
	return nil
}

func (m *UpdateContainerResourcesRequest) GetWindows() *WindowsContainerResources {
	if m != nil {
		return m.Windows
	}
	return nil
}

func (m *UpdateContainerResourcesRequest) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *ExecSyncRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}

func (m *ExecSyncRequest) GetCmd() []string {
	if m != nil {
		return m.Cmd
	}
	return nil
}

func (m *ExecSyncRequest) GetTimeout() int64 {
	if m != nil {
		return m.Timeout
	}
	return 0
}

func (m *ExecSyncResponse) GetStdout() []byte {
	if m != nil {
		return m.Stdout
	}
	return nil
}

func (m *ExecSyncResponse) GetStderr() []byte {
	if m != nil {
		return m.Stderr
	}
	return nil
}

func (m *ExecSyncResponse) GetExitCode() int32 {
	if m != nil {
		return m.ExitCode
	}
	return 0
}

func (m *ExecRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}

func (m *ExecRequest) GetCmd() []string {
	if m != nil {
		return m.Cmd
	}
	return nil
}

func (m *ExecRequest) GetTty() bool {
	if m != nil {
		return m.Tty
	}
	return false
}

func (m *ExecRequest) GetStdin() bool {
	if m != nil {
		return m.Stdin
	}
	return false
}

func (m *ExecRequest) GetStdout() bool {
	if m != nil {
		return m.Stdout
	}
	return false
}

func (m *ExecRequest) GetStderr() bool {
	if m != nil {
		return m.Stderr
	}
	return false
}

func (m *ExecResponse) GetUrl() string {
	if m != nil {
		return m.Url
	}
	return ""
}

func (m *AttachRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}

func (m *AttachRequest) GetStdin() bool {
	if m != nil {
		return m.Stdin
	}
	return false
}

func (m *AttachRequest) GetTty() bool {
	if m != nil {
		return m.Tty
	}
	return false
}

func (m *AttachRequest) GetStdout() bool {
	if m != nil {
		return m.Stdout
	}
	return false
}

func (m *AttachRequest) GetStderr() bool {
	if m != nil {
		return m.Stderr
	}
	return false
}

func (m *AttachResponse) GetUrl() string {
	if m != nil {
		return m.Url
	}
	return ""
}

func (m *PortForwardRequest) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *PortForwardRequest) GetPort() []int32 {
	if m != nil {
		return m.Port
	}
	return nil
}

func (m *PortForwardResponse) GetUrl() string {
	if m != nil {
		return m.Url
	}
	return ""
}

func (m *ImageFilter) GetImage() *ImageSpec {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *ListImagesRequest) GetFilter() *ImageFilter {
	if m != nil {
		return m.Filter
	}
	return nil
}

func (m *Image) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *Image) GetRepoTags() []string {
	if m != nil {
		return m.RepoTags
	}
	return nil
}

func (m *Image) GetRepoDigests() []string {
	if m != nil {
		return m.RepoDigests
	}
	return nil
}

func (m *Image) GetSize_() uint64 {
	if m != nil {
		return m.Size_
	}
	return 0
}

func (m *Image) GetUid() *Int64Value {
	if m != nil {
		return m.Uid
	}
	return nil
}

func (m *Image) GetUsername() string {
	if m != nil {
		return m.Username
	}
	return ""
}

func (m *Image) GetSpec() *ImageSpec {
	if m != nil {
		return m.Spec
	}
	return nil
}

func (m *ListImagesResponse) GetImages() []*Image {
	if m != nil {
		return m.Images
	}
	return nil
}

func (m *ImageStatusRequest) GetImage() *ImageSpec {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *ImageStatusRequest) GetVerbose() bool {
	if m != nil {
		return m.Verbose
	}
	return false
}

func (m *ImageStatusResponse) GetImage() *Image {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *ImageStatusResponse) GetInfo() map[string]string {
	if m != nil {
		return m.Info
	}
	return nil
}

func (m *AuthConfig) GetUsername() string {
	if m != nil {
		return m.Username
	}
	return ""
}

func (m *AuthConfig) GetPassword() string {
	if m != nil {
		return m.Password
	}
	return ""
}

func (m *AuthConfig) GetAuth() string {
	if m != nil {
		return m.Auth
	}
	return ""
}

func (m *AuthConfig) GetServerAddress() string {
	if m != nil {
		return m.ServerAddress
	}
	return ""
}

func (m *AuthConfig) GetIdentityToken() string {
	if m != nil {
		return m.IdentityToken
	}
	return ""
}

func (m *AuthConfig) GetRegistryToken() string {
	if m != nil {
		return m.RegistryToken
	}
	return ""
}

func (m *PullImageRequest) GetImage() *ImageSpec {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *PullImageRequest) GetAuth() *AuthConfig {
	if m != nil {
		return m.Auth
	}
	return nil
}

func (m *PullImageRequest) GetSandboxConfig() *PodSandboxConfig {
	if m != nil {
		return m.SandboxConfig
	}
	return nil
}

func (m *PullImageResponse) GetImageRef() string {
	if m != nil {
		return m.ImageRef
	}
	return ""
}

func (m *RemoveImageRequest) GetImage() *ImageSpec {
	if m != nil {
		return m.Image
	}
	return nil
}

func (m *NetworkConfig) GetPodCidr() string {
	if m != nil {
		return m.PodCidr
	}
	return ""
}

func (m *RuntimeConfig) GetNetworkConfig() *NetworkConfig {
	if m != nil {
		return m.NetworkConfig
	}
	return nil
}

func (m *UpdateRuntimeConfigRequest) GetRuntimeConfig() *RuntimeConfig {
	if m != nil {
		return m.RuntimeConfig
	}
	return nil
}

func (m *RuntimeCondition) GetType() string {
	if m != nil {
		return m.Type
	}
	return ""
}

func (m *RuntimeCondition) GetStatus() bool {
	if m != nil {
		return m.Status
	}
	return false
}

func (m *RuntimeCondition) GetReason() string {
	if m != nil {
		return m.Reason
	}
	return ""
}

func (m *RuntimeCondition) GetMessage() string {
	if m != nil {
		return m.Message
	}
	return ""
}

func (m *RuntimeStatus) GetConditions() []*RuntimeCondition {
	if m != nil {
		return m.Conditions
	}
	return nil
}

func (m *StatusRequest) GetVerbose() bool {
	if m != nil {
		return m.Verbose
	}
	return false
}

func (m *StatusResponse) GetStatus() *RuntimeStatus {
	if m != nil {
		return m.Status
	}
	return nil
}

func (m *StatusResponse) GetInfo() map[string]string {
	if m != nil {
		return m.Info
	}
	return nil
}

func (m *UInt64Value) GetValue() uint64 {
	if m != nil {
		return m.Value
	}
	return 0
}

func (m *FilesystemIdentifier) GetMountpoint() string {
	if m != nil {
		return m.Mountpoint
	}
	return ""
}

func (m *FilesystemUsage) GetTimestamp() int64 {
	if m != nil {
		return m.Timestamp
	}
	return 0
}

func (m *FilesystemUsage) GetFsId() *FilesystemIdentifier {
	if m != nil {
		return m.FsId
	}
	return nil
}

func (m *FilesystemUsage) GetUsedBytes() *UInt64Value {
	if m != nil {
		return m.UsedBytes
	}
	return nil
}

func (m *FilesystemUsage) GetInodesUsed() *UInt64Value {
	if m != nil {
		return m.InodesUsed
	}
	return nil
}

func (m *ImageFsInfoResponse) GetImageFilesystems() []*FilesystemUsage {
	if m != nil {
		return m.ImageFilesystems
	}
	return nil
}

func (m *ContainerStatsRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}

func (m *ContainerStatsResponse) GetStats() *ContainerStats {
	if m != nil {
		return m.Stats
	}
	return nil
}

func (m *ListContainerStatsRequest) GetFilter() *ContainerStatsFilter {
	if m != nil {
		return m.Filter
	}
	return nil
}

func (m *ContainerStatsFilter) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *ContainerStatsFilter) GetPodSandboxId() string {
	if m != nil {
		return m.PodSandboxId
	}
	return ""
}

func (m *ContainerStatsFilter) GetLabelSelector() map[string]string {
	if m != nil {
		return m.LabelSelector
	}
	return nil
}

func (m *ListContainerStatsResponse) GetStats() []*ContainerStats {
	if m != nil {
		return m.Stats
	}
	return nil
}

func (m *ContainerAttributes) GetId() string {
	if m != nil {
		return m.Id
	}
	return ""
}

func (m *ContainerAttributes) GetMetadata() *ContainerMetadata {
	if m != nil {
		return m.Metadata
	}
	return nil
}

func (m *ContainerAttributes) GetLabels() map[string]string {
	if m != nil {
		return m.Labels
	}
	return nil
}

func (m *ContainerAttributes) GetAnnotations() map[string]string {
	if m != nil {
		return m.Annotations
	}
	return nil
}

func (m *ContainerStats) GetAttributes() *ContainerAttributes {
	if m != nil {
		return m.Attributes
	}
	return nil
}

func (m *ContainerStats) GetCpu() *CpuUsage {
	if m != nil {
		return m.Cpu
	}
	return nil
}

func (m *ContainerStats) GetMemory() *MemoryUsage {
	if m != nil {
		return m.Memory
	}
	return nil
}

func (m *ContainerStats) GetWritableLayer() *FilesystemUsage {
	if m != nil {
		return m.WritableLayer
	}
	return nil
}

func (m *CpuUsage) GetTimestamp() int64 {
	if m != nil {
		return m.Timestamp
	}
	return 0
}

func (m *CpuUsage) GetUsageCoreNanoSeconds() *UInt64Value {
	if m != nil {
		return m.UsageCoreNanoSeconds
	}
	return nil
}

func (m *CpuUsage) GetUsageNanoCores() *UInt64Value {
	if m != nil {
		return m.UsageNanoCores
	}
	return nil
}

func (m *MemoryUsage) GetTimestamp() int64 {
	if m != nil {
		return m.Timestamp
	}
	return 0
}

func (m *MemoryUsage) GetWorkingSetBytes() *UInt64Value {
	if m != nil {
		return m.WorkingSetBytes
	}
	return nil
}

func (m *MemoryUsage) GetAvailableBytes() *UInt64Value {
	if m != nil {
		return m.AvailableBytes
	}
	return nil
}

func (m *MemoryUsage) GetUsageBytes() *UInt64Value {
	if m != nil {
		return m.UsageBytes
	}
	return nil
}

func (m *MemoryUsage) GetRssBytes() *UInt64Value {
	if m != nil {
		return m.RssBytes
	}
	return nil
}

func (m *MemoryUsage) GetPageFaults() *UInt64Value {
	if m != nil {
		return m.PageFaults
	}
	return nil
}

func (m *MemoryUsage) GetMajorPageFaults() *UInt64Value {
	if m != nil {
		return m.MajorPageFaults
	}
	return nil
}

func (m *ReopenContainerLogRequest) GetContainerId() string {
	if m != nil {
		return m.ContainerId
	}
	return ""
}
