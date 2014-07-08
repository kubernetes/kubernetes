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
	"github.com/fsouza/go-dockerclient"
)

// Common string formats
// ---------------------
// Many fields in this API have formatting requirements.  The commonly used
// formats are defined here.
//
// C_IDENTIFIER:  This is a string that conforms the definition of an "identifier"
//     in the C language.  This is captured by the following regex:
//         [A-Za-z_][A-Za-z0-9_]*
//     This defines the format, but not the length restriction, which should be
//     specified at the definition of any field of this type.
//
// DNS_LABEL:  This is a string that conforms to the definition of a "label"
//     in RFCs 1035 and 1123.  This is captured by the following regex:
//         [a-z0-9]([-a-z0-9]*[a-z0-9])?
//     This defines the format, but not the length restriction, which should be
//     specified at the definition of any field of this type.
//
//  DNS_SUBDOMAIN:  This is a string that conforms to the definition of a
//      "subdomain" in RFCs 1035 and 1123.  This is captured by the following regex:
//         [a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*
//     or more simply:
//         DNS_LABEL(\.DNS_LABEL)*
//     This defines the format, but not the length restriction, which should be
//     specified at the definition of any field of this type.

// ContainerManifest corresponds to the Container Manifest format, documented at:
// https://developers.google.com/compute/docs/containers/container_vms#container_manifest
// This is used as the representation of Kubernetes workloads.
type ContainerManifest struct {
	// Required: This must be a supported version string, such as "v1beta1".
	Version string `yaml:"version" json:"version"`
	// Required: This must be a DNS_SUBDOMAIN, 255 characters or less.
	ID         string      `yaml:"id" json:"id"`
	Volumes    []Volume    `yaml:"volumes" json:"volumes"`
	Containers []Container `yaml:"containers" json:"containers"`
}

// Volume represents a named volume in a pod that may be accessed by any containers in the pod.
type Volume struct {
	// Required: This must be a DNS_LABEL, 63 characters or less.  Each volume in a pod
	// must have a unique name.
	Name string `yaml:"name" json:"name"`
}

// Port represents a network port in a single container
type Port struct {
	// Optional: If specified, this must be a DNS_LABEL, 63 characters or less.  Each
	// container in a pod must have a unique name.
	Name string `yaml:"name,omitempty" json:"name,omitempty"`
	// Optional: Defaults to ContainerPort.
	HostPort int `yaml:"hostPort,omitempty" json:"hostPort,omitempty"`
	// Required: This must be a valid port number, 0 < x < 65536.
	ContainerPort int `yaml:"containerPort" json:"containerPort"`
	// Optional: Defaults to "TCP".
	Protocol string `yaml:"protocol,omitempty" json:"protocol,omitempty"`
}

// VolumeMount describes a mounting of a Volume within a container
type VolumeMount struct {
	// Required: This must match the Name of a Volume [above].
	Name string `yaml:"name" json:"name"`
	// Optional: Defaults to false (read-write).
	ReadOnly bool `yaml:"readOnly,omitempty" json:"readOnly,omitempty"`
	// Required.
	MountPath string `yaml:"mountPath,omitempty" json:"mountPath,omitempty"`
	// One of: "LOCAL" (local volume) or "HOST" (external mount from the host). Default: LOCAL.
	MountType string `yaml:"mountType,omitempty" json:"mountType,omitempty"`
}

// EnvVar represents an environment variable present in a Container
type EnvVar struct {
	// Required: This must be a C_IDENTIFIER.
	// Exactly one of the following must be set.  If both are set, prefer Name.
	// DEPRECATED: EnvVar.Key will be removed in a future version of the API.
	Name string `yaml:"name" json:"name"`
	Key  string `yaml:"key,omitempty" json:"key,omitempty"`
	// Optional: defaults to "".
	Value string `yaml:"value,omitempty" json:"value,omitempty"`
}

// Container represents a single container that is expected to be run on the host.
type Container struct {
	// Required: This must be a DNS_LABEL, 63 characters or less.  Each container in a
	// pod must have a unique name.
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
	CPU          int           `yaml:"cpu,omitempty" json:"cpu,omitempty"`
	VolumeMounts []VolumeMount `yaml:"volumeMounts,omitempty" json:"volumeMounts,omitempty"`
}

// Percentile represents a pair which contains a percentage from 0 to 100 and
// its corresponding value.
type Percentile struct {
	Percentage int    `json:"percentage,omitempty"`
	Value      uint64 `json:"value,omitempty"`
}

// ContainerStats represents statistical information of a container
type ContainerStats struct {
	CpuUsagePercentiles    []Percentile `json:"cpu_usage_percentiles,omitempty"`
	MemoryUsagePercentiles []Percentile `json:"memory_usage_percentiles,omitempty"`
	MaxMemoryUsage         uint64       `json:"max_memory_usage,omitempty"`
}

// Event is the representation of an event logged to etcd backends
type Event struct {
	Event     string             `json:"event,omitempty"`
	Manifest  *ContainerManifest `json:"manifest,omitempty"`
	Container *Container         `json:"container,omitempty"`
	Timestamp int64              `json:"timestamp"`
}

// The below types are used by kube_client and api_server.

// JSONBase is shared by all objects sent to, or returned from the client
type JSONBase struct {
	Kind              string `json:"kind,omitempty" yaml:"kind,omitempty"`
	ID                string `json:"id,omitempty" yaml:"id,omitempty"`
	CreationTimestamp string `json:"creationTimestamp,omitempty" yaml:"creationTimestamp,omitempty"`
	SelfLink          string `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
	ResourceVersion   uint64 `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
}

// PodStatus represents a status of a pod.
type PodStatus string

// These are the valid statuses of pods.
const (
	PodRunning PodStatus = "Running"
	PodPending PodStatus = "Pending"
	PodStopped PodStatus = "Stopped"
)

// PodInfo contains one entry for every container with available info.
type PodInfo map[string]docker.Container

// PodState is the state of a pod, used as either input (desired state) or output (current state)
type PodState struct {
	Manifest ContainerManifest `json:"manifest,omitempty" yaml:"manifest,omitempty"`
	Status   PodStatus         `json:"status,omitempty" yaml:"status,omitempty"`
	Host     string            `json:"host,omitempty" yaml:"host,omitempty"`
	HostIP   string            `json:"hostIP,omitempty" yaml:"hostIP,omitempty"`

	// The key of this map is the *name* of the container within the manifest; it has one
	// entry per container in the manifest. The value of this map is currently the output
	// of `docker inspect`. This output format is *not* final and should not be relied
	// upon.
	// TODO: Make real decisions about what our info should look like.
	Info PodInfo `json:"info,omitempty" yaml:"info,omitempty"`
}

// PodList is a list of Pods.
type PodList struct {
	JSONBase `json:",inline" yaml:",inline"`
	Items    []Pod `json:"items" yaml:"items,omitempty"`
}

// Pod is a collection of containers, used as either input (create, update) or as output (list, get)
type Pod struct {
	JSONBase     `json:",inline" yaml:",inline"`
	Labels       map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
	DesiredState PodState          `json:"desiredState,omitempty" yaml:"desiredState,omitempty"`
	CurrentState PodState          `json:"currentState,omitempty" yaml:"currentState,omitempty"`
}

// ReplicationControllerState is the state of a replication controller, either input (create, update) or as output (list, get)
type ReplicationControllerState struct {
	Replicas        int               `json:"replicas" yaml:"replicas"`
	ReplicaSelector map[string]string `json:"replicaSelector,omitempty" yaml:"replicaSelector,omitempty"`
	PodTemplate     PodTemplate       `json:"podTemplate,omitempty" yaml:"podTemplate,omitempty"`
}

// ReplicationControllerList is a collection of replication controllers.
type ReplicationControllerList struct {
	JSONBase `json:",inline" yaml:",inline"`
	Items    []ReplicationController `json:"items,omitempty" yaml:"items,omitempty"`
}

// ReplicationController represents the configuration of a replication controller
type ReplicationController struct {
	JSONBase     `json:",inline" yaml:",inline"`
	DesiredState ReplicationControllerState `json:"desiredState,omitempty" yaml:"desiredState,omitempty"`
	Labels       map[string]string          `json:"labels,omitempty" yaml:"labels,omitempty"`
}

// PodTemplate holds the information used for creating pods
type PodTemplate struct {
	DesiredState PodState          `json:"desiredState,omitempty" yaml:"desiredState,omitempty"`
	Labels       map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}

// ServiceList holds a list of services
type ServiceList struct {
	JSONBase `json:",inline" yaml:",inline"`
	Items    []Service `json:"items" yaml:"items"`
}

// Service is a named abstraction of software service (for example, mysql) consisting of local port
// (for example 3306) that the proxy listens on, and the selector that determines which pods
// will answer requests sent through the proxy.
type Service struct {
	JSONBase `json:",inline" yaml:",inline"`
	Port     int `json:"port,omitempty" yaml:"port,omitempty"`

	// This service's labels.
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`

	// This service will route traffic to pods having labels matching this selector.
	Selector                   map[string]string `json:"selector,omitempty" yaml:"selector,omitempty"`
	CreateExternalLoadBalancer bool              `json:"createExternalLoadBalancer,omitempty" yaml:"createExternalLoadBalancer,omitempty"`
}

// Endpoints is a collection of endpoints that implement the actual service, for example:
// Name: "mysql", Endpoints: ["10.10.1.1:1909", "10.10.2.2:8834"]
type Endpoints struct {
	Name      string
	Endpoints []string
}

// Minion is a worker node in Kubernetenes.
// The name of the minion according to etcd is in JSONBase.ID.
type Minion struct {
	JSONBase `json:",inline" yaml:",inline"`
	// Queried from cloud provider, if available.
	HostIP string `json:"hostIP,omitempty" yaml:"hostIP,omitempty"`
}

// MinionList is a list of minions.
type MinionList struct {
	JSONBase `json:",inline" yaml:",inline"`
	Items    []Minion `json:"minions,omitempty" yaml:"minions,omitempty"`
}

// Status is a return value for calls that don't return other objects.
// Arguably, this could go in apiserver, but I'm including it here so clients needn't
// import both.
type Status struct {
	JSONBase `json:",inline" yaml:",inline"`
	// One of: "success", "failure", "working" (for operations not yet completed)
	// TODO: if "working", include an operation identifier so final status can be
	// checked.
	Status string `json:"status,omitempty" yaml:"status,omitempty"`
	// Details about the status. May be an error description or an
	// operation number for later polling.
	Details string `json:"details,omitempty" yaml:"details,omitempty"`
	// Suggested HTTP return code for this status, 0 if not set.
	Code int `json:"code,omitempty" yaml:"code,omitempty"`
}

// Values of Status.Status
const (
	StatusSuccess = "success"
	StatusFailure = "failure"
	StatusWorking = "working"
)

// ServerOp is an operation delivered to API clients.
type ServerOp struct {
	JSONBase `yaml:",inline" json:",inline"`
}

// ServerOpList is a list of operations, as delivered to API clients.
type ServerOpList struct {
	JSONBase `yaml:",inline" json:",inline"`
	Items    []ServerOp `yaml:"items,omitempty" json:"items,omitempty"`
}
