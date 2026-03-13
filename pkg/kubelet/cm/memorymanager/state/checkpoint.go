/*
Copyright 2020 The Kubernetes Authors.

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

package state

import (
	"encoding/json"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpoint{}

// MemoryManagerCheckpoint struct is used to store memory/pod assignments in a checkpoint
type MemoryManagerCheckpoint struct {
	PolicyName   string                     `json:"policyName"`
	MachineState NUMANodeMap                `json:"machineState"`
	Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
	Checksum     checksum.Checksum          `json:"checksum"`
}

// NewMemoryManagerCheckpoint returns an instance of Checkpoint
func NewMemoryManagerCheckpoint() *MemoryManagerCheckpoint {
	//nolint:staticcheck // unexported-type-in-api user-facing error message
	return &MemoryManagerCheckpoint{
		Entries:      ContainerMemoryAssignments{},
		MachineState: NUMANodeMap{},
	}
}

// MarshalCheckpoint returns marshalled checkpoint
func (mp *MemoryManagerCheckpoint) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	mp.Checksum = 0
	mp.Checksum = checksum.New(mp)
	return json.Marshal(*mp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint
func (mp *MemoryManagerCheckpoint) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, mp)
}

// VerifyChecksum verifies that current checksum of checkpoint is valid
func (mp *MemoryManagerCheckpoint) VerifyChecksum() error {
	ck := mp.Checksum
	mp.Checksum = 0
	err := ck.Verify(mp)
	mp.Checksum = ck
	return err
}
