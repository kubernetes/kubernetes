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
	"fmt"
	"hash/fnv"
	"strings"

	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

var _ checkpointmanager.Checkpoint = &DRAManagerCheckpoint{}

const checkpointVersion = "v1"

// DRAManagerCheckpoint struct is used to store pod dynamic resources assignments in a checkpoint
type DRAManagerCheckpoint struct {
	Version  string             `json:"version"`
	Entries  ClaimInfoStateList `json:"entries,omitempty"`
	Checksum checksum.Checksum  `json:"checksum"`
}

// DraManagerCheckpoint struct is an old implementation of the DraManagerCheckpoint
type DRAManagerCheckpointWithoutResourceHandles struct {
	Version  string                                   `json:"version"`
	Entries  ClaimInfoStateListWithoutResourceHandles `json:"entries,omitempty"`
	Checksum checksum.Checksum                        `json:"checksum"`
}

// List of claim info to store in checkpoint
type ClaimInfoStateList []ClaimInfoState

// List of claim info to store in checkpoint
// TODO: remove in Beta
type ClaimInfoStateListWithoutResourceHandles []ClaimInfoStateWithoutResourceHandles

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
	if err == errors.ErrCorruptCheckpoint {
		// Verify with old structs without ResourceHandles field
		// TODO: remove in Beta
		err = verifyChecksumWithoutResourceHandles(dc, ck)
	}
	dc.Checksum = ck
	return err
}

// verifyChecksumWithoutResourceHandles is a helper function that verifies checksum of the
// checkpoint in the old format, without ResourceHandles field.
// TODO: remove in Beta.
func verifyChecksumWithoutResourceHandles(dc *DRAManagerCheckpoint, checkSum checksum.Checksum) error {
	entries := ClaimInfoStateListWithoutResourceHandles{}
	for _, entry := range dc.Entries {
		entries = append(entries, ClaimInfoStateWithoutResourceHandles{
			DriverName: entry.DriverName,
			ClassName:  entry.ClassName,
			ClaimUID:   entry.ClaimUID,
			ClaimName:  entry.ClaimName,
			Namespace:  entry.Namespace,
			PodUIDs:    entry.PodUIDs,
			CDIDevices: entry.CDIDevices,
		})
	}
	oldcheckpoint := &DRAManagerCheckpointWithoutResourceHandles{
		Version:  checkpointVersion,
		Entries:  entries,
		Checksum: 0,
	}
	// Calculate checksum for old checkpoint
	object := dump.ForHash(oldcheckpoint)
	object = strings.Replace(object, "DRAManagerCheckpointWithoutResourceHandles", "DRAManagerCheckpoint", 1)
	object = strings.Replace(object, "ClaimInfoStateListWithoutResourceHandles", "ClaimInfoStateList", 1)
	hash := fnv.New32a()
	fmt.Fprintf(hash, "%v", object)
	if checkSum != checksum.Checksum(hash.Sum32()) {
		return errors.ErrCorruptCheckpoint
	}
	return nil
}
