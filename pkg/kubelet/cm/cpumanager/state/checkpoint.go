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

	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

var _ checkpointmanager.Checkpoint = &CPUManagerCheckpointV1{}
var _ checkpointmanager.Checkpoint = &CPUManagerCheckpointV2{}
var _ checkpointmanager.Checkpoint = &CPUManagerCheckpoint{}

// CPUManagerCheckpoint struct is used to store cpu/pod assignments in a checkpoint in v2 format
type CPUManagerCheckpoint struct {
	PolicyName    string                       `json:"policyName"`
	DefaultCPUSet string                       `json:"defaultCpuSet"`
	Entries       map[string]map[string]string `json:"entries,omitempty"`
	Checksum      checksum.Checksum            `json:"checksum"`
}

// CPUManagerCheckpointV1 struct is used to store cpu/pod assignments in a checkpoint in v1 format
type CPUManagerCheckpointV1 struct {
	PolicyName    string            `json:"policyName"`
	DefaultCPUSet string            `json:"defaultCpuSet"`
	Entries       map[string]string `json:"entries,omitempty"`
	Checksum      checksum.Checksum `json:"checksum"`
}

// CPUManagerCheckpointV2 struct is used to store cpu/pod assignments in a checkpoint in v2 format
type CPUManagerCheckpointV2 = CPUManagerCheckpoint

// NewCPUManagerCheckpoint returns an instance of Checkpoint
func NewCPUManagerCheckpoint() *CPUManagerCheckpoint {
	//nolint:staticcheck // unexported-type-in-api user-facing error message
	return newCPUManagerCheckpointV2()
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
	cp.Checksum = checksum.New(cp)
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
	fmt.Fprintf(hash, "%v", object)
	if cp.Checksum != checksum.Checksum(hash.Sum32()) {
		return errors.ErrCorruptCheckpoint
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
	err := ck.Verify(cp)
	cp.Checksum = ck
	return err
}
