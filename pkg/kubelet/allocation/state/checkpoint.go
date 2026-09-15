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
	"encoding/base64"
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

const (
	// checkpointVersionV2 is the current checkpoint format version storing base64-encoded protobuf PodList.
	checkpointVersionV2 = "v2"
)

var _ checkpointmanager.Checkpoint = &Checkpoint{}

// Checkpoint represents a structure to store pod resource allocation checkpoint data.
type Checkpoint struct {
	// Version is the checkpoint format version (e.g. "v2")
	Version string `json:"version,omitempty"`
	// Data is a serialized and base64-encoded PodList
	Data string `json:"data"`
	// Checksum is a checksum of Data
	Checksum checksum.Checksum `json:"checksum"`
}

// NewCheckpoint creates a new checkpoint containing the serialized PodList.
func NewCheckpoint(podList *v1.PodList) (*Checkpoint, error) {
	if podList == nil {
		podList = &v1.PodList{}
	}
	protoBytes, err := podList.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to marshal PodList to protobuf for checkpointing: %w", err)
	}
	data := base64.StdEncoding.EncodeToString(protoBytes)
	cp := &Checkpoint{
		Version:  checkpointVersionV2,
		Data:     data,
		Checksum: checksum.New(data),
	}
	return cp, nil
}

// GetPodList deserializes the base64-encoded protobuf PodList from checkpoint data.
func (cp *Checkpoint) getPodList() (*v1.PodList, error) {
	protoBytes, err := base64.StdEncoding.DecodeString(cp.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64 protobuf data: %w", err)
	}
	var podList v1.PodList
	if err := podList.Unmarshal(protoBytes); err != nil {
		return nil, fmt.Errorf("failed to unmarshal protobuf PodList: %w", err)
	}
	return &podList, nil
}

func (cp *Checkpoint) MarshalCheckpoint() ([]byte, error) {
	return json.Marshal(cp)
}

// UnmarshalCheckpoint unmarshals checkpoint from  JSON
func (cp *Checkpoint) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

func (cp *Checkpoint) VerifyChecksum() error {
	return cp.Checksum.Verify(cp.Data)
}
