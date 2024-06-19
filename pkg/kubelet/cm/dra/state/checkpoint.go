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

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

var _ checkpointmanager.Checkpoint = &DRAManagerCheckpoint{}

const checkpointVersion = "v1"

// DRAManagerCheckpoint struct is used to store pod dynamic resources assignments in a checkpoint
type DRAManagerCheckpoint struct {
	Version  string             `json:"version"`
	Entries  ClaimInfoStateList `json:"entries,omitempty"`
	Checksum checksum.Checksum  `json:"checksum"`
}

// List of claim info to store in checkpoint
type ClaimInfoStateList []ClaimInfoState

// NewDRAManagerCheckpoint returns an instance of Checkpoint
func NewDRAManagerCheckpoint() *DRAManagerCheckpoint {
	return &DRAManagerCheckpoint{
		Version: checkpointVersion,
		Entries: ClaimInfoStateList{},
	}
}

// MarshalCheckpoint returns marshalled checkpoint
func (dc *DRAManagerCheckpoint) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	dc.Checksum = 0
	dc.Checksum = checksum.New(dc)
	return json.Marshal(*dc)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint
func (dc *DRAManagerCheckpoint) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, dc)
}

// VerifyChecksum verifies that current checksum of checkpoint is valid
func (dc *DRAManagerCheckpoint) VerifyChecksum() error {
	ck := dc.Checksum
	dc.Checksum = 0
	err := ck.Verify(dc)
	dc.Checksum = ck
	return err
}
