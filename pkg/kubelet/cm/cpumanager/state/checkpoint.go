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
	"hash/fnv"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

var _ checkpointmanager.Checkpoint = &CPUManagerCheckpoint{}

// CPUManagerCheckpoint struct is used to store cpu/pod assignments in a checkpoint
type CPUManagerCheckpoint struct {
	PolicyName    string
	DefaultCPUSet string
	Entries       map[string]string
	Checksum      uint64
}

// NewCPUManagerCheckpoint returns an instance of Checkpoint
func NewCPUManagerCheckpoint() *CPUManagerCheckpoint {
	return &CPUManagerCheckpoint{Entries: make(map[string]string)}
}

// MarshalCheckpoint returns marshalled checkpoint
func (cp *CPUManagerCheckpoint) MarshalCheckpoint() ([]byte, error) {
	return json.Marshal(*cp)
}

// UnmarshalCheckpoint tries to unmarshal passed bytes to checkpoint
func (cp *CPUManagerCheckpoint) UnmarshalCheckpoint(blob []byte) error {
	if err := json.Unmarshal(blob, cp); err != nil {
		return err
	}
	if cp.Checksum != cp.GetChecksum() {
		return errors.ErrCorruptCheckpoint
	}
	return nil
}

// GetChecksum returns calculated checksum of checkpoint
func (cp *CPUManagerCheckpoint) GetChecksum() uint64 {
	orig := cp.Checksum
	cp.Checksum = 0
	hash := fnv.New32a()
	hashutil.DeepHashObject(hash, *cp)
	cp.Checksum = orig
	return uint64(hash.Sum32())
}

// UpdateChecksum calculates and updates checksum of the checkpoint
func (cp *CPUManagerCheckpoint) UpdateChecksum() {
	cp.Checksum = cp.GetChecksum()
}
