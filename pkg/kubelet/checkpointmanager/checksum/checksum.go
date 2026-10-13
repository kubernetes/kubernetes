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

package checksum

import (
	"hash/fnv"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

// Checksum is the data to be stored as checkpoint
type Checksum uint64

// Verify verifies that passed checksum is same as calculated checksum
func (cs Checksum) Verify(data interface{}) error {
	actualCS := New(data)
	if cs != actualCS {
		return &errors.CorruptCheckpointError{ActualCS: uint64(actualCS), ExpectedCS: uint64(cs)}
	}
	return nil
}

// New returns the Checksum of checkpoint data
func New(data interface{}) Checksum {
	return Checksum(getChecksum(data))
}

// Get returns calculated checksum of checkpoint data
func getChecksum(data interface{}) uint64 {
	hash := fnv.New32a()
	hashutil.DeepHashObject(hash, data)
	return uint64(hash.Sum32())
}
