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

package container

import (
	"io"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type PodSandboxID string

// PodSandboxManager provides basic operations to create/delete and examine the
// PodSandboxes. These methods should either return an error or block until the
// operation succeeds.
type PodSandboxManager interface {
	// Create creates a sandbox based on the given config, and returns the
	// the new sandbox.
	Create(config *PodSandboxConfig) (PodSandboxID, error)
	// Stop stops the sandbox by its ID. If there are any running
	// containers in the sandbox, they will be terminated as a side-effect.
	Stop(id PodSandboxID) error
	// Delete deletes the sandbox by its ID. If there are any running
	// containers in the sandbox, they will be deleted as a side-effect.
	Delete(id PodSandboxID) error
	// List lists existing sandboxes, filtered by the given PodSandboxFilter.
	List(filter PodSandboxFilter) ([]PodSandboxListItem, error)
	// Status gets the status of the sandbox by ID.
	Status(id PodSandboxID) (PodSandboxStatus, error)
}

// PodSandboxConfig holds all the required and optional fields for creating a
// sandbox.
type PodSandboxConfig struct {
	// Name is the name of the sandbox. The string should conform to
	// [a-zA-Z0-9_-]+.
	Name string
	// Hostname is the hostname of the sandbox.
	Hostname string
	// DNSOptions sets the DNS options for the sandbox.
	DNSOptions DNSOptions
	// PortMappings lists the port mappings for the sandbox.
	PortMappings []PortMapping
	// Resources specifies the resource limits for the sandbox (i.e., the
	// aggregate cpu/memory resources limits of all containers).
	// Note: On a Linux host, kubelet will create a pod-level cgroup and pass
	// it as the cgroup parent for the PodSandbox. For some runtimes, this is
	// sufficient. For others, e.g., hypervisor-based runtimes, explicit
	// resource limits for the sandbox are needed at creation time.
	Resources PodSandboxResources
	// Path to the directory on the host in which container log files are
	// stored.
	// By default the Log of a container going into the LogDirectory will be
	// hooked up to STDOUT and STDERR. However, the LogDirectory may contain
	// binary log files with structured logging data from the individual
	// containers. For example, the files might be newline separated JSON
	// structured logs, systemd-journald journal files, gRPC trace files, etc.
	// E.g.,
	//     PodSandboxConfig.LogDirectory = `/var/log/pods/<podUID>/`
	//     ContainerConfig.LogPath = `containerName_Instance#.log`
	//
	// WARNING: Log management and how kubelet should interface with the
	// container logs are under active discussion in
	// https://issues.k8s.io/24677. There *may* be future change of direction
	// for logging as the discussion carries on.
	LogDirectory string
	// Labels are key value pairs that may be used to scope and select
	// individual resources.
	Labels Labels
	// Annotations is an unstructured key value map that may be set by external
	// tools to store and retrieve arbitrary metadata.
	Annotations map[string]string

	// Linux contains configurations specific to Linux hosts.
	Linux *LinuxPodSandboxConfig
}

// Labels are key value pairs that may be used to scope and select individual
// resources.
// Label keys are of the form:
//     label-key ::= prefixed-name | name
//     prefixed-name ::= prefix '/' name
//     prefix ::= DNS_SUBDOMAIN
//     name ::= DNS_LABEL
type Labels map[string]string

// LinuxPodSandboxConfig holds platform-specific configuraions for Linux
// host platforms and Linux-based containers.
type LinuxPodSandboxConfig struct {
	// CgroupParent is the parent cgroup of the sandbox. The cgroupfs style
	// syntax will be used, but the container runtime can convert it to systemd
	// semantices if needed.
	CgroupParent string
	// NamespaceOptions contains configurations for the sandbox's namespaces.
	// This will be used only if the PodSandbox uses namespace for isolation.
	NamespaceOptions NamespaceOptions
}

// NamespaceOptions provides options for Linux namespaces.
type NamespaceOptions struct {
	// HostNetwork uses the host's network namespace.
	HostNetwork bool
	// HostPID uses the host's pid namesapce.
	HostPID bool
	// HostIPC uses the host's ipc namespace.
	HostIPC bool
}

// DNSOptions specifies the DNS servers and search domains.
type DNSOptions struct {
	// Servers is a list of DNS servers of the cluster.
	Servers []string
	// Searches is a list of DNS search domains of the cluster.
	Searches []string
}

type PodSandboxState string

const (
	// PodSandboxReady means the sandbox is functioning properly.
	PodSandboxReady PodSandboxState = "Ready"
	// PodSandboxInNotReady means the sandbox is not functioning properly.
	PodSandboxNotReady PodSandboxState = "NotReady"
)

// PodSandboxFilter is used to filter a list of PodSandboxes.
type PodSandboxFilter struct {
	// Name of the sandbox.
	Name *string
	// ID of the sandbox.
	ID *PodSandboxID
	// State of the sandbox.
	State *PodSandboxState
	// LabelSelector to select matches.
	// Only api.MatchLabels is supported for now and the requirements
	// are ANDed. MatchExpressions is not supported yet.
	LabelSelector unversioned.LabelSelector
}

// PodSandboxListItem contains minimal information about a sandbox.
type PodSandboxListItem struct {
	ID    PodSandboxID
	State PodSandboxState
	// Labels are key value pairs that may be used to scope and select individual resources.
	Labels Labels
}

// PodSandboxStatus contains the status of the PodSandbox.
type PodSandboxStatus struct {
	// ID of the sandbox.
	ID PodSandboxID
	// State of the sandbox.
	State PodSandboxState
	// Network contains network status if network is handled by the runtime.
	Network *PodSandboxNetworkStatus
	// Status specific to a Linux sandbox.
	Linux *LinuxPodSandboxStatus
	// Labels are key value pairs that may be used to scope and select individual resources.
	Labels Labels
	// Annotations is an unstructured key value map.
	Annotations map[string]string
}

// PodSandboxNetworkStatus is the status of the network for a PodSandbox.
type PodSandboxNetworkStatus struct {
	IPs []string
}

// Namespaces contains paths to the namespaces.
type Namespaces struct {
	// Network is the path to the network namespace.
	Network string
}

// LinuxSandBoxStatus contains status specific to Linux sandboxes.
type LinuxPodSandboxStatus struct {
	// Namespaces contains paths to the sandbox's namespaces.
	Namespaces *Namespaces
}

// PodSandboxResources contains the CPU/memory resource requirements.
type PodSandboxResources struct {
	// CPU resource requirement.
	CPU resource.Quantity
	// Memory resource requirement.
	Memory resource.Quantity
}

// This is to distinguish with existing ContainerID type, which includes a
// runtime type prefix (e.g., docker://). We may rename this later.
type RawContainerID string

// ContainerRuntime provides methods for container lifecycle operations, as
// well as listing or inspecting existing containers. These methods should
// either return an error or block until the operation succeeds.
type ContainerRuntime interface {
	// Create creates a container in the sandbox, and returns the ID
	// of the created container.
	Create(config *ContainerConfig, sandboxConfig *PodSandboxConfig, sandboxID PodSandboxID) (RawContainerID, error)
	// Start starts a created container.
	Start(id RawContainerID) error
	// Stop stops a running container with a grace period (i.e., timeout).
	Stop(id RawContainerID, timeout int) error
	// Remove removes the container.
	Remove(id RawContainerID) error
	// List lists the existing containers that match the ContainerFilter.
	// The returned list should only include containers previously created
	// by this ContainerRuntime.
	List(filter ContainerFilter) ([]ContainerListItem, error)
	// Status returns the status of the container.
	Status(id RawContainerID) (RawContainerStatus, error)
	// Exec executes a command in the container.
	Exec(id RawContainerID, cmd []string, streamOpts StreamOptions) error
}

// ContainerListItem provides the runtime information for a container returned
// by List().
type ContainerListItem struct {
	// The ID of the container, used by the container runtime to identify
	// a container.
	ID ContainerID
	// The name of the container, which should be the same as specified by
	// api.Container.
	Name string
	// Reference to the image in use. For most runtimes, this should be an
	// image ID.
	ImageRef string
	// State is the state of the container.
	State ContainerState
	// Labels are key value pairs that may be used to scope and select individual resources.
	Labels Labels
}

type ContainerConfig struct {
	// Name of the container. The string should conform to [a-zA-Z0-9_-]+.
	Name string
	// Image to use.
	Image ImageSpec
	// Command to execute (i.e., entrypoint for docker)
	Command []string
	// Args for the Command (i.e., command for docker)
	Args []string
	// Current working directory of the command.
	WorkingDir string
	// List of environment variable to set in the container
	Env []KeyValue
	// Mounts specifies mounts for the container
	Mounts []Mount
	// Labels are key value pairs that may be used to scope and select individual resources.
	Labels Labels
	// Annotations is an unstructured key value map that may be set by external
	// tools to store and retrieve arbitrary metadata.
	Annotations map[string]string
	// Privileged runs the container in the privileged mode.
	Privileged bool
	// ReadOnlyRootFS sets the root filesystem of the container to be
	// read-only.
	ReadOnlyRootFS bool
	// Path relative to PodSandboxConfig.LogDirectory for container to store
	// the log (STDOUT and STDERR) on the host.
	// E.g.,
	//     PodSandboxConfig.LogDirectory = `/var/log/pods/<podUID>/`
	//     ContainerConfig.LogPath = `containerName_Instance#.log`
	//
	// WARNING: Log management and how kubelet should interface with the
	// container logs are under active discussion in
	// https://issues.k8s.io/24677. There *may* be future change of direction
	// for logging as the discussion carries on.
	LogPath string

	// Variables for interactive containers, these have very specialized
	// use-cases (e.g. debugging).
	// TODO: Determine if we need to continue supporting these fields that are
	// part of Kubernetes's Container Spec.
	STDIN     bool
	STDINONCE bool
	TTY       bool

	// Linux contains configuration specific to Linux containers.
	Linux *LinuxContainerConfig
}

// RawContainerStatus represents the status of a container.
type RawContainerStatus struct {
	// ID of the container.
	ID ContainerID
	// Name of the container.
	Name string
	// Status of the container.
	State ContainerState
	// Creation time of the container.
	CreatedAt unversioned.Time
	// Start time of the container.
	StartedAt unversioned.Time
	// Finish time of the container.
	FinishedAt unversioned.Time
	// Exit code of the container.
	ExitCode int
	// Reference to the image in use. For most runtimes, this should be an
	// image ID.
	ImageRef string
	// Labels are key value pairs that may be used to scope and select individual resources.
	Labels Labels
	// Annotations is an unstructured key value map.
	Annotations map[string]string
	// A brief CamelCase string explains why container is in such a status.
	Reason string
}

// LinuxContainerConfig contains platform-specific configuration for
// Linux-based containers.
type LinuxContainerConfig struct {
	// Resources specification for the container.
	Resources *LinuxContainerResources
	// Capabilities to add or drop.
	Capabilities *api.Capabilities
	// SELinux is the SELinux context to be applied.
	SELinux *api.SELinuxOptions
	// TODO: Add support for seccomp.
}

// LinuxContainerResources specifies Linux specific configuration for
// resources.
// TODO: Consider using Resources from opencontainers/runtime-spec/specs-go
// directly.
type LinuxContainerResources struct {
	// CPU CFS (Completely Fair Scheduler) period
	CPUPeriod *int64
	// CPU CFS (Completely Fair Scheduler) quota
	CPUQuota *int64
	// CPU shares (relative weight vs. other containers)
	CPUShares *int64
	// Memory limit in bytes
	MemoryLimitInBytes *int64
	// OOMScoreAdj adjusts the oom-killer score.
	OOMScoreAdj *int64
}

// ContainerFilter is used to filter containers.
type ContainerFilter struct {
	// Name of the container.
	Name *string
	// ID of the container.
	ID *RawContainerID
	// State of the contianer.
	State *ContainerState
	// ID of the PodSandbox.
	PodSandboxID *PodSandboxID
	// LabelSelector to select matches.
	// Only api.MatchLabels is supported for now and the requirements
	// are ANDed. MatchExpressions is not supported yet.
	LabelSelector unversioned.LabelSelector
}

type StreamOptions struct {
	TTY          bool
	InputStream  io.Reader
	OutputStream io.Writer
	ErrorStream  io.Writer
}

// KeyValue represents a key-value pair.
type KeyValue struct {
	Key   string
	Value string
}

// ImageService offers basic image operations.
type ImageService interface {
	// List lists the existing images.
	List() ([]Image, error)
	// Pull pulls an image with authentication config. The PodSandboxConfig is
	// passed so that the image service can charge the resources used for
	// pulling to a sepcific pod.
	Pull(image ImageSpec, auth AuthConfig, sandboxConfig *PodSandboxConfig) error
	// Remove removes an image.
	Remove(image ImageSpec) error
	// Status returns the status of an image.
	Status(image ImageSpec) (Image, error)
}

// AuthConfig contains authorization information for connecting to a registry.
// TODO: This is copied from docker's Authconfig. We should re-evaluate to
// support other registries.
type AuthConfig struct {
	Username      string
	Password      string
	Auth          string
	ServerAddress string
	// IdentityToken is used to authenticate the user and get
	// an access token for the registry.
	IdentityToken string
	// RegistryToken is a bearer token to be sent to a registry
	RegistryToken string
}

// TODO: Add ContainerMetricsGetter and ImageMetricsGetter.
