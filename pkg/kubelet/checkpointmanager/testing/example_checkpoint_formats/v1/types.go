/*
Copyright 2018 The Kubernetes Authors.

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
	"encoding/json"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

type protocol string

// PortMapping is the port mapping configurations of a sandbox.
type PortMapping struct {
	// protocol of the port mapping.
	Protocol *protocol
	// Port number within the container.
	ContainerPort *int32
	// Port number on the host.
	HostPort *int32
}

// Data contains all types of data that can be stored in the checkpoint.
type Data struct {
	PortMappings []*PortMapping `json:"port_mappings,omitempty"`
	HostNetwork  bool           `json:"host_network,omitempty"`
}

// CheckpointData is a sample example structure to be used in test cases for checkpointing
type CheckpointData struct {
	Version  string
	Name     string
	Data     *Data
	Checksum checksum.Checksum
}

// MarshalCheckpoint tries to marshal the CheckpointData into JSON data.
func (cp *CheckpointData) MarshalCheckpoint() ([]byte, error) {
	cp.Checksum = checksum.New(*cp.Data)
	return json.Marshal(*cp)
}

// UnmarshalCheckpoint tries to unmarshal the passed JSON data into CheckpointData.
func (cp *CheckpointData) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// VerifyChecksum verifies that CheckpointData's checksum is same as the checksum calculated by its data.
func (cp *CheckpointData) VerifyChecksum() error {
	return cp.Checksum.Verify(*cp.Data)
}
