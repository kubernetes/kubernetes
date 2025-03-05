/*
Copyright 2021 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

var _ checkpointmanager.Checkpoint = &Checkpoint{}

type PodResourceAllocationInfo struct {
	AllocationEntries map[types.UID]map[string]v1.ResourceRequirements `json:"allocationEntries,omitempty"`
}

// Checkpoint represents a structure to store pod resource allocation checkpoint data
type Checkpoint struct {
	// Data is a serialized PodResourceAllocationInfo
	Data string `json:"data"`
	// Checksum is a checksum of Data
	Checksum checksum.Checksum `json:"checksum"`
}

// NewCheckpoint creates a new checkpoint from a list of claim info states
func NewCheckpoint(allocations *PodResourceAllocationInfo) (*Checkpoint, error) {

	serializedAllocations, err := json.Marshal(allocations)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize allocations for checkpointing: %w", err)
	}

	cp := &Checkpoint{
		Data: string(serializedAllocations),
	}
	cp.Checksum = checksum.New(cp.Data)
	return cp, nil
}

func (cp *Checkpoint) MarshalCheckpoint() ([]byte, error) {
	return json.Marshal(cp)
}

// UnmarshalCheckpoint unmarshals checkpoint from JSON
func (cp *Checkpoint) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// VerifyChecksum verifies that current checksum
// of checkpointed Data is valid
func (cp *Checkpoint) VerifyChecksum() error {
	return cp.Checksum.Verify(cp.Data)
}

// GetPodResourceAllocationInfo returns Pod Resource Allocation info states from checkpoint
func (cp *Checkpoint) GetPodResourceAllocationInfo() (*PodResourceAllocationInfo, error) {
	var data PodResourceAllocationInfo
	if err := json.Unmarshal([]byte(cp.Data), &data); err != nil {
		return nil, err
	}

	return &data, nil
}
