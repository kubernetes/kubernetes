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

package state

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"strings"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	"k8s.io/utils/dump"
)

// CPUManagerCheckpointV4 serializes checkpoint data to a string, following the
// same approach used in the allocation manager (see pkg/kubelet/allocation/state/checkpoint.go).
// The embedded V3 version is for keeping forward compatibility.
var _ checkpointmanager.Checkpoint = &CPUManagerCheckpointV1{}
var _ checkpointmanager.Checkpoint = &CPUManagerCheckpointV2{}
var _ checkpointmanager.Checkpoint = &CPUManagerCheckpointV3{}
var _ checkpointmanager.Checkpoint = &CPUManagerCheckpointV4{}
var _ checkpointmanager.Checkpoint = &CPUManagerCheckpoint{}

// CPUManagerCheckpointData struct is used to store CPU/pod assignments, which is part of a checkpoint in v4 format
type CPUManagerCheckpointData struct {
	PolicyName    string                       `json:"policyName"`
	DefaultCPUSet string                       `json:"defaultCpuSet"`
	Entries       map[string]map[string]string `json:"entries,omitempty"`
	PodEntries    PodCPUAssignments            `json:"podEntries,omitempty"`
}

// CPUManagerCheckpoint represents a structure to store CPU manager checkpoint data.
// Dual-checksum format for backward compatibility.
// The Data string with DataChecksum is the authoritative V4 payload.
type CPUManagerCheckpoint struct {
	// Backward compatibility
	CPUManagerCheckpointV3 `json:",inline"`

	// Data is a serialized CPUManagerCheckpointData
	Data string `json:"data"`
	// DataChecksum is a checksum of Data string
	DataChecksum checksum.Checksum `json:"dataChecksum"`

	// CheckpointData holds actual data, not serialized directly
	CheckpointData CPUManagerCheckpointData `json:"-"`
}

// CPUManagerCheckpoint struct is used to store CPU/pod assignments in a checkpoint in v4 format
type CPUManagerCheckpointV4 = CPUManagerCheckpoint

// CPUManagerCheckpointV3 struct is used to store CPU/pod assignments in a checkpoint in v3 format
type CPUManagerCheckpointV3 struct {
	PolicyName    string                       `json:"policyName"`
	DefaultCPUSet string                       `json:"defaultCpuSet"`
	Entries       map[string]map[string]string `json:"entries,omitempty"`
	PodEntries    PodCPUAssignments            `json:"podEntries,omitempty"`
	Checksum      checksum.Checksum            `json:"checksum"`
}

// CPUManagerCheckpointV2 struct is used to store CPU/pod assignments in a checkpoint in v2 format
type CPUManagerCheckpointV2 struct {
	PolicyName    string                       `json:"policyName"`
	DefaultCPUSet string                       `json:"defaultCpuSet"`
	Entries       map[string]map[string]string `json:"entries,omitempty"`
	Checksum      checksum.Checksum            `json:"checksum"`
}

// CPUManagerCheckpointV1 struct is used to store CPU/pod assignments in a checkpoint in v1 format
type CPUManagerCheckpointV1 struct {
	PolicyName    string            `json:"policyName"`
	DefaultCPUSet string            `json:"defaultCpuSet"`
	Entries       map[string]string `json:"entries,omitempty"`
	Checksum      checksum.Checksum `json:"checksum"`
}

// NewCPUManagerCheckpoint returns an instance of Checkpoint
func newCPUManagerCheckpoint() *CPUManagerCheckpoint {
	//nolint:staticcheck // unexported-type-in-api user-facing error message
	return newCPUManagerCheckpointV4()
}

func newCPUManagerCheckpointV1() *CPUManagerCheckpointV1 {
	return &CPUManagerCheckpointV1{
		Entries: make(map[string]string),
	}
}

func newCPUManagerCheckpointV2() *CPUManagerCheckpointV2 {
	return &CPUManagerCheckpointV2{
		Entries: make(map[string]map[string]string),
	}
}

func newCPUManagerCheckpointV3() *CPUManagerCheckpointV3 {
	return &CPUManagerCheckpointV3{
		Entries:    make(map[string]map[string]string),
		PodEntries: make(PodCPUAssignments),
	}
}

func newCPUManagerCheckpointV4() *CPUManagerCheckpointV4 {
	return &CPUManagerCheckpoint{
		CPUManagerCheckpointV3: CPUManagerCheckpointV3{
			Entries:    make(map[string]map[string]string),
			PodEntries: make(PodCPUAssignments),
		},
		CheckpointData: CPUManagerCheckpointData{
			Entries:    make(map[string]map[string]string),
			PodEntries: make(PodCPUAssignments),
		},
	}
}

// MarshalCheckpoint returns marshalled checkpoint in v1 format
func (cp *CPUManagerCheckpointV1) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	cp.Checksum = 0
	cp.Checksum = checksum.New(cp)
	return json.Marshal(*cp)
}

// MarshalCheckpoint returns marshalled checkpoint in v2 format
func (cp *CPUManagerCheckpointV2) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	cp.Checksum = 0

	// In order to preserve rollback compatibility when the feature gate is disabled,
	// we must generate a checksum using the legacy struct name "CPUManagerCheckpoint"
	// instead of "CPUManagerCheckpointV2". Older Kubelets do not have the string
	// replacement logic and expect the original struct name.
	object := dump.ForHash(cp)
	object = strings.Replace(object, "CPUManagerCheckpointV2", "CPUManagerCheckpoint", 1)
	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	cp.Checksum = checksum.Checksum(hash.Sum32())

	return json.Marshal(*cp)
}

// MarshalCheckpoint returns marshalled checkpoint in v3 format
func (cp *CPUManagerCheckpointV3) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	cp.Checksum = 0

	// In order to preserve rollback compatibility we must generate a checksum using
	// the legacy struct name "CPUManagerCheckpoint" instead of "CPUManagerCheckpointV3".
	// Older Kubelets do not have the string replacement logic
	// and expect the original struct name.
	object := dump.ForHash(cp)
	object = strings.Replace(object, "CPUManagerCheckpointV3", "CPUManagerCheckpoint", 1)
	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	cp.Checksum = checksum.Checksum(hash.Sum32())

	return json.Marshal(*cp)
}

// MarshalCheckpoint returns marshalled checkpoint in v4 format with dual checksums.
func (cp *CPUManagerCheckpoint) MarshalCheckpoint() ([]byte, error) {
	data, err := json.Marshal(cp.CheckpointData)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize cpu manager checkpoint data: %w", err)
	}
	cp.Data = string(data)
	cp.DataChecksum = checksum.New(cp.Data)

	cp.PolicyName = cp.CheckpointData.PolicyName
	cp.DefaultCPUSet = cp.CheckpointData.DefaultCPUSet
	cp.Entries = cp.CheckpointData.Entries
	cp.PodEntries = cp.CheckpointData.PodEntries

	// For forward compatibility, clear checksum from the legacy field.
	// This way it can be read by older kubelets as V2 or V3.
	cp.Checksum = 0

	return json.Marshal(*cp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint in v1 format
func (cp *CPUManagerCheckpointV1) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint in v2 format
func (cp *CPUManagerCheckpointV2) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint in v3 format
func (cp *CPUManagerCheckpointV3) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint in v4 format
func (cp *CPUManagerCheckpoint) UnmarshalCheckpoint(blob []byte) error {
	if err := json.Unmarshal(blob, cp); err != nil {
		return fmt.Errorf("failed to deserialize cpu manager checkpoint: %w", err)
	}
	if err := json.Unmarshal([]byte(cp.Data), &cp.CheckpointData); err != nil {
		return fmt.Errorf("failed to deserialize cpu manager checkpoint data: %w", err)
	}
	return nil
}

// VerifyChecksum verifies that current checksum of checkpoint is valid in v1 format
func (cp *CPUManagerCheckpointV1) VerifyChecksum() error {
	if cp.Checksum == 0 {
		// accept empty checksum for compatibility with old file backend
		return nil
	}

	ck := cp.Checksum
	cp.Checksum = 0
	object := dump.ForHash(cp)
	object = strings.Replace(object, "CPUManagerCheckpointV1", "CPUManagerCheckpoint", 1)
	cp.Checksum = ck

	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	actualCS := checksum.Checksum(hash.Sum32())
	if cp.Checksum != actualCS {
		return &errors.CorruptCheckpointError{
			ActualCS:   uint64(actualCS),
			ExpectedCS: uint64(cp.Checksum),
		}
	}

	return nil
}

// VerifyChecksum verifies that current checksum of checkpoint is valid in v2 format
func (cp *CPUManagerCheckpointV2) VerifyChecksum() error {
	if cp.Checksum == 0 {
		// accept empty checksum for compatibility with old file backend
		return nil
	}

	ck := cp.Checksum
	cp.Checksum = 0

	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) {
		err := ck.Verify(cp)
		if err == nil {
			cp.Checksum = ck
			return nil
		}
		// If the standard verification failed, it could be a legacy checkpoint.
		// Fallback to the legacy verification method, when V3 is latest.
	}

	object := dump.ForHash(cp)
	object = strings.Replace(object, "CPUManagerCheckpointV2", "CPUManagerCheckpoint", 1)
	cp.Checksum = ck

	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	actualCS := checksum.Checksum(hash.Sum32())
	if cp.Checksum != actualCS {
		return &errors.CorruptCheckpointError{
			ActualCS:   uint64(actualCS),
			ExpectedCS: uint64(cp.Checksum),
		}
	}
	return nil
}

// VerifyChecksum verifies that current checksum of checkpoint is valid in v3 format
func (cp *CPUManagerCheckpointV3) VerifyChecksum() error {
	ck := cp.Checksum
	cp.Checksum = 0
	err := ck.Verify(cp)
	if err == nil {
		cp.Checksum = ck
		return nil
	}

	object := dump.ForHash(cp)
	object = strings.Replace(object, "CPUManagerCheckpointV3", "CPUManagerCheckpoint", 1)
	cp.Checksum = ck

	hash := fnv.New32a()
	_, _ = fmt.Fprintf(hash, "%v", object)
	actualCS := checksum.Checksum(hash.Sum32())
	if cp.Checksum != actualCS {
		return &errors.CorruptCheckpointError{
			ActualCS:   uint64(actualCS),
			ExpectedCS: uint64(cp.Checksum),
		}
	}
	return nil
}

// VerifyChecksum verifies that current checksum of checkpoint is valid in v4 format.
// DataChecksum is the authoritative checksum over Data.
func (cp *CPUManagerCheckpoint) VerifyChecksum() error {
	return cp.DataChecksum.Verify(cp.Data)
}
