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

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

var _ checkpointmanager.Checkpoint = &PodResourceAllocationCheckpoint{}

// PodResourceAllocationCheckpoint is used to store resources allocated to a pod in checkpoint
type PodResourceAllocationCheckpoint struct {
	AllocationEntries   map[string]map[string]v1.ResourceList `json:"allocationEntries,omitempty"`
	ResizeStatusEntries map[string]v1.PodResizeStatus         `json:"resizeStatusEntries,omitempty"`
	Checksum            checksum.Checksum                     `json:"checksum"`
}

// NewPodResourceAllocationCheckpoint returns an instance of Checkpoint
func NewPodResourceAllocationCheckpoint() *PodResourceAllocationCheckpoint {
	//lint:ignore unexported-type-in-api user-facing error message
	return &PodResourceAllocationCheckpoint{
		AllocationEntries:   make(map[string]map[string]v1.ResourceList),
		ResizeStatusEntries: make(map[string]v1.PodResizeStatus),
	}
}

// MarshalCheckpoint returns marshalled checkpoint
func (prc *PodResourceAllocationCheckpoint) MarshalCheckpoint() ([]byte, error) {
	// make sure checksum wasn't set before so it doesn't affect output checksum
	prc.Checksum = 0
	prc.Checksum = checksum.New(prc)
	return json.Marshal(*prc)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint
func (prc *PodResourceAllocationCheckpoint) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, prc)
}

// VerifyChecksum verifies that current checksum of checkpoint is valid
func (prc *PodResourceAllocationCheckpoint) VerifyChecksum() error {
	ck := prc.Checksum
	prc.Checksum = 0
	err := ck.Verify(prc)
	prc.Checksum = ck
	return err
}
