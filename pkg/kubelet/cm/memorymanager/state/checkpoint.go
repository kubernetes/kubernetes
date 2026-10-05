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
	"fmt"
	"hash/fnv"
	"strings"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	"k8s.io/utils/dump"
)

// MemoryManagerCheckpointV3 serializes checkpoint data to a string, following the
// same approach used in the allocation manager (see pkg/kubelet/allocation/state/checkpoint.go).
// The embedded V1 version is for keeping forward compatibility.
var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpointV1{}
var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpointV2{}
var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpointV3{}
var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpoint{}

// MemoryManagerCheckpointData struct is used to store memory/pod assignments, which is part of a checkpoint in v3 format
type MemoryManagerCheckpointData struct {
	PolicyName   string                     `json:"policyName"`
	MachineState NUMANodeMap                `json:"machineState"`
	Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
	PodEntries   PodMemoryAssignments       `json:"podEntries,omitempty"`
}

// MemoryManagerCheckpoint represents a structure to store memory/pod assignments checkpoint data.
// Dual-checksum format for backward compatibility.
// The Data string with DataChecksum is the authoritative V3 payload.
type MemoryManagerCheckpoint struct {
	// Backward compatibility
	// The embedded V1 checkpoint must not be enhanced and should be removed as soon as the oldest supported version understands V3.
	MemoryManagerCheckpointV1 `json:",inline"`

	// Data is a serialized MemoryManagerCheckpointData
	Data string `json:"data"`
	// DataChecksum is a checksum of Data string
	DataChecksum checksum.Checksum `json:"dataChecksum"`

	// CheckpointData holds actual data, not serialized directly
	CheckpointData MemoryManagerCheckpointData `json:"-"`
}

// MemoryManagerCheckpointV3 struct is used to store memory/pod assignments in a checkpoint in v3 format
type MemoryManagerCheckpointV3 = MemoryManagerCheckpoint

// MemoryManagerCheckpointV2 struct is used to store memory/pod assignments in a checkpoint in v2 format
type MemoryManagerCheckpointV2 struct {
	PolicyName   string                     `json:"policyName"`
	MachineState NUMANodeMap                `json:"machineState"`
	Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
	PodEntries   PodMemoryAssignments       `json:"podEntries,omitempty"`
	Checksum     checksum.Checksum          `json:"checksum"`
}

// MemoryManagerCheckpointV1 struct is used to store memory/pod assignments in a checkpoint in v1 format
type MemoryManagerCheckpointV1 struct {
	PolicyName   string                     `json:"policyName"`
	MachineState NUMANodeMap                `json:"machineState"`
	Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
	Checksum     checksum.Checksum          `json:"checksum"`
}

// NewMemoryManagerCheckpoint returns an instance of Checkpoint
func newMemoryManagerCheckpoint() *MemoryManagerCheckpoint {
	//nolint:staticcheck // unexported-type-in-api user-facing error message
	return newMemoryManagerCheckpointV3()
}

func newMemoryManagerCheckpointV1() *MemoryManagerCheckpointV1 {
	return &MemoryManagerCheckpointV1{
		Entries:      ContainerMemoryAssignments{},
		MachineState: NUMANodeMap{},
	}
}

func newMemoryManagerCheckpointV2() *MemoryManagerCheckpointV2 {
	return &MemoryManagerCheckpointV2{
		Entries:      ContainerMemoryAssignments{},
		PodEntries:   PodMemoryAssignments{},
		MachineState: NUMANodeMap{},
	}
}

func newMemoryManagerCheckpointV3() *MemoryManagerCheckpointV3 {
	return &MemoryManagerCheckpoint{
		MemoryManagerCheckpointV1: MemoryManagerCheckpointV1{
			Entries:      ContainerMemoryAssignments{},
			MachineState: NUMANodeMap{},
		},
		CheckpointData: MemoryManagerCheckpointData{
			Entries:      ContainerMemoryAssignments{},
			PodEntries:   PodMemoryAssignments{},
			MachineState: NUMANodeMap{},
		},
	}
}

// MarshalCheckpoint returns marshalled checkpoint in v1 format
func (mp *MemoryManagerCheckpointV1) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	mp.Checksum = 0

	// In order to preserve rollback compatibility,
	// we must generate a checksum using the legacy struct name "MemoryManagerCheckpoint"
	// instead of "MemoryManagerCheckpointV1". Older Kubelets do not have the string
	// replacement logic and expect the original struct name.
	object := dump.ForHash(mp)
	object = strings.Replace(object, "MemoryManagerCheckpointV1", "MemoryManagerCheckpoint", 1)
	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	mp.Checksum = checksum.Checksum(hash.Sum32())

	return json.Marshal(*mp)
}

// MarshalCheckpoint returns marshalled checkpoint in v2 format
func (mp *MemoryManagerCheckpointV2) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	mp.Checksum = 0

	// In order to preserve rollback compatibility,
	// we must generate a checksum using the legacy struct name "MemoryManagerCheckpoint"
	// instead of "MemoryManagerCheckpointV2". Older Kubelets do not have the string
	// replacement logic and expect the original struct name.
	object := dump.ForHash(mp)
	object = strings.Replace(object, "MemoryManagerCheckpointV2", "MemoryManagerCheckpoint", 1)
	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	mp.Checksum = checksum.Checksum(hash.Sum32())

	return json.Marshal(*mp)
}

// MarshalCheckpoint returns marshalled checkpoint in v3 format
func (mp *MemoryManagerCheckpointV3) MarshalCheckpoint() ([]byte, error) {
	data, err := json.Marshal(mp.CheckpointData)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize memory manager checkpoint data: %w", err)
	}
	mp.Data = string(data)
	mp.DataChecksum = checksum.New(mp.Data)

	mp.PolicyName = mp.CheckpointData.PolicyName
	mp.MachineState = mp.CheckpointData.MachineState
	mp.Entries = mp.CheckpointData.Entries

	mp.Checksum = 0
	object := dump.ForHash(&mp.MemoryManagerCheckpointV1)
	object = strings.Replace(object, "MemoryManagerCheckpointV1", "MemoryManagerCheckpoint", 1)
	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	mp.Checksum = checksum.Checksum(hash.Sum32())

	return json.Marshal(*mp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint in v1 format
func (mp *MemoryManagerCheckpointV1) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, mp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint in v2 format
func (mp *MemoryManagerCheckpointV2) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, mp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint in v3 format
func (mp *MemoryManagerCheckpointV3) UnmarshalCheckpoint(blob []byte) error {
	if err := json.Unmarshal(blob, mp); err != nil {
		return fmt.Errorf("failed to deserialize memory manager checkpoint: %w", err)
	}
	if err := json.Unmarshal([]byte(mp.Data), &mp.CheckpointData); err != nil {
		return fmt.Errorf("failed to deserialize memory manager checkpoint data: %w", err)
	}
	return nil
}

// VerifyChecksum verifies that current checksum of checkpoint is valid in v1 format
func (mp *MemoryManagerCheckpointV1) VerifyChecksum() error {
	if mp.Checksum == 0 {
		// accept empty checksum for compatibility with old file backend
		return nil
	}

	ck := mp.Checksum
	mp.Checksum = 0
	object := dump.ForHash(mp)
	object = strings.Replace(object, "MemoryManagerCheckpointV1", "MemoryManagerCheckpoint", 1)
	mp.Checksum = ck

	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	actualCS := checksum.Checksum(hash.Sum32())
	if mp.Checksum != actualCS {
		return &errors.CorruptCheckpointError{
			ActualCS:   uint64(actualCS),
			ExpectedCS: uint64(mp.Checksum),
		}
	}

	return nil
}

// VerifyChecksum verifies that current checksum of checkpoint is valid in v2 format
func (mp *MemoryManagerCheckpointV2) VerifyChecksum() error {
	ck := mp.Checksum
	mp.Checksum = 0
	object := dump.ForHash(mp)
	object = strings.Replace(object, "MemoryManagerCheckpointV2", "MemoryManagerCheckpoint", 1)
	mp.Checksum = ck

	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	actualCS := checksum.Checksum(hash.Sum32())
	if mp.Checksum != actualCS {
		return &errors.CorruptCheckpointError{
			ActualCS:   uint64(actualCS),
			ExpectedCS: uint64(mp.Checksum),
		}
	}

	return nil
}

// VerifyChecksum verifies that current checksum of checkpoint is valid in v3 format.
// DataChecksum is the authoritative checksum over Data.
func (mp *MemoryManagerCheckpoint) VerifyChecksum() error {
	return mp.DataChecksum.Verify(mp.Data)
}
