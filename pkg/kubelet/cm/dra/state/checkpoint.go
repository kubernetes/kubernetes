/*
Copyright 2023 The Kubernetes Authors.

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
	"hash/crc32"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

const (
	CheckpointAPIGroup   = "checkpoint.dra.kubelet.k8s.io"
	CheckpointKind       = "DRACheckpoint"
	CheckpointAPIVersion = CheckpointAPIGroup + "/v1"
)

// Checkpoint represents a structure to store DRA checkpoint data
type Checkpoint struct {
	// Data is a JSON serialized checkpoint data
	Data string
	// Checksum is a checksum of Data
	Checksum uint32
}

type CheckpointData struct {
	metav1.TypeMeta
	ClaimInfoStateList ClaimInfoStateList
}

// NewCheckpoint creates a new checkpoint from a list of claim info states
func NewCheckpoint(data ClaimInfoStateList) (*Checkpoint, error) {
	cpData := &CheckpointData{
		TypeMeta: metav1.TypeMeta{
			Kind:       CheckpointKind,
			APIVersion: CheckpointAPIVersion,
		},
		ClaimInfoStateList: data,
	}

	cpDataBytes, err := json.Marshal(cpData)
	if err != nil {
		return nil, err
	}

	cp := &Checkpoint{
		Data:     string(cpDataBytes),
		Checksum: crc32.ChecksumIEEE(cpDataBytes),
	}

	return cp, nil
}

// MarshalCheckpoint marshals checkpoint to JSON
func (cp *Checkpoint) MarshalCheckpoint() ([]byte, error) {
	return json.Marshal(cp)
}

// UnmarshalCheckpoint unmarshals checkpoint from JSON
// and verifies its data checksum
func (cp *Checkpoint) UnmarshalCheckpoint(blob []byte) error {
	if err := json.Unmarshal(blob, cp); err != nil {
		return err
	}

	// verify checksum
	if err := cp.VerifyChecksum(); err != nil {
		return err
	}

	return nil
}

// VerifyChecksum verifies that current checksum
// of checkpointed Data is valid
func (cp *Checkpoint) VerifyChecksum() error {
	expectedCS := crc32.ChecksumIEEE([]byte(cp.Data))
	if expectedCS != cp.Checksum {
		return &errors.CorruptCheckpointError{ActualCS: uint64(cp.Checksum), ExpectedCS: uint64(expectedCS)}
	}
	return nil
}

// GetClaimInfoStateList returns list of claim info states from checkpoint
func (cp *Checkpoint) GetClaimInfoStateList() (ClaimInfoStateList, error) {
	var data CheckpointData
	if err := json.Unmarshal([]byte(cp.Data), &data); err != nil {
		return nil, err
	}

	return data.ClaimInfoStateList, nil
}
