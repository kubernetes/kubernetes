// +build !dockerless

/*
Copyright 2017 The Kubernetes Authors.

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

package dockershim

import (
	"encoding/json"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

const (
	// default directory to store pod sandbox checkpoint files
	sandboxCheckpointDir = "sandbox"
	protocolTCP          = Protocol("tcp")
	protocolUDP          = Protocol("udp")
	protocolSCTP         = Protocol("sctp")
	schemaVersion        = "v1"
)

// ContainerCheckpoint provides the interface for process container's checkpoint data
type ContainerCheckpoint interface {
	checkpointmanager.Checkpoint
	GetData() (string, string, string, []*PortMapping, bool)
}

// Protocol is the type of port mapping protocol
type Protocol string

// PortMapping is the port mapping configurations of a sandbox.
type PortMapping struct {
	// Protocol of the port mapping.
	Protocol *Protocol `json:"protocol,omitempty"`
	// Port number within the container.
	ContainerPort *int32 `json:"container_port,omitempty"`
	// Port number on the host.
	HostPort *int32 `json:"host_port,omitempty"`
	// Host ip to expose.
	HostIP string `json:"host_ip,omitempty"`
}

// CheckpointData contains all types of data that can be stored in the checkpoint.
type CheckpointData struct {
	PortMappings []*PortMapping `json:"port_mappings,omitempty"`
	HostNetwork  bool           `json:"host_network,omitempty"`
}

// PodSandboxCheckpoint is the checkpoint structure for a sandbox
type PodSandboxCheckpoint struct {
	// Version of the pod sandbox checkpoint schema.
	Version string `json:"version"`
	// Pod name of the sandbox. Same as the pod name in the Pod ObjectMeta.
	Name string `json:"name"`
	// Pod namespace of the sandbox. Same as the pod namespace in the Pod ObjectMeta.
	Namespace string `json:"namespace"`
	// Data to checkpoint for pod sandbox.
	Data *CheckpointData `json:"data,omitempty"`
	// Checksum is calculated with fnv hash of the checkpoint object with checksum field set to be zero
	Checksum checksum.Checksum `json:"checksum"`
}

// NewPodSandboxCheckpoint inits a PodSandboxCheckpoint with the given args
func NewPodSandboxCheckpoint(namespace, name string, data *CheckpointData) ContainerCheckpoint {
	return &PodSandboxCheckpoint{
		Version:   schemaVersion,
		Namespace: namespace,
		Name:      name,
		Data:      data,
	}
}

// MarshalCheckpoint encodes the PodSandboxCheckpoint instance to a json object
func (cp *PodSandboxCheckpoint) MarshalCheckpoint() ([]byte, error) {
	cp.Checksum = checksum.New(*cp.Data)
	return json.Marshal(*cp)
}

// UnmarshalCheckpoint decodes the blob data to the PodSandboxCheckpoint instance
func (cp *PodSandboxCheckpoint) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// VerifyChecksum verifies whether the PodSandboxCheckpoint's data checksum is
// the same as calculated checksum
func (cp *PodSandboxCheckpoint) VerifyChecksum() error {
	return cp.Checksum.Verify(*cp.Data)
}

// GetData gets the PodSandboxCheckpoint's version and some net information
func (cp *PodSandboxCheckpoint) GetData() (string, string, string, []*PortMapping, bool) {
	return cp.Version, cp.Name, cp.Namespace, cp.Data.PortMappings, cp.Data.HostNetwork
}
