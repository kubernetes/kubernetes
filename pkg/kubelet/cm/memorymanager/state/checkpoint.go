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

	"k8s.io/apimachinery/pkg/util/dump"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpointV1{}
var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpointV2{}
var _ checkpointmanager.Checkpoint = &MemoryManagerCheckpoint{}

// MemoryManagerCheckpoint struct is used to store memory/pod assignments in a checkpoint in v2 format
type MemoryManagerCheckpoint struct {
	PolicyName   string                     `json:"policyName"`
	MachineState NUMANodeMap                `json:"machineState"`
	Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
	PodEntries   PodMemoryAssignments       `json:"podEntries,omitempty"`
	Checksum     checksum.Checksum          `json:"checksum"`
}

// MemoryManagerCheckpointV2 struct is used to store memory/pod assignments in a checkpoint in v2 format
type MemoryManagerCheckpointV2 = MemoryManagerCheckpoint

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
	return newMemoryManagerCheckpointV2()
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

// MarshalCheckpoint returns marshalled checkpoint in v1 format
func (mp *MemoryManagerCheckpointV1) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	mp.Checksum = 0
	mp.Checksum = checksum.New(mp)
	return json.Marshal(*mp)
}

// MarshalCheckpoint returns marshalled checkpoint in v2 format
func (mp *MemoryManagerCheckpointV2) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	mp.Checksum = 0
	mp.Checksum = checksum.New(mp)
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

// VerifyChecksum verifies that current checksum of checkpoint is valid in v1 format
func (mp *MemoryManagerCheckpointV1) VerifyChecksum() error {
	if mp.Checksum == 0 {
		// accept empty checksum for compatibility with old file backend
		return nil
	}

	ck := mp.Checksum
	mp.Checksum = 0

	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		err := ck.Verify(mp)
		if err == nil {
			mp.Checksum = ck
			return nil
		}
		// If the standard verification failed, it could be a legacy checkpoint.
		// Fallback to the legacy verification method, when V2 is latest.
	}

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
func (mp *MemoryManagerCheckpoint) VerifyChecksum() error {
	ck := mp.Checksum
	mp.Checksum = 0
	err := ck.Verify(mp)
	mp.Checksum = ck
	return err
}
