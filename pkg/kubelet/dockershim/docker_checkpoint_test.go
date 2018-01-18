/*
Copyright 2017 The Kubernetes Authors.

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

package dockershim

import (
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	utilstore "k8s.io/kubernetes/pkg/kubelet/dockershim/testing"
)

func NewTestPersistentCheckpointHandler() CheckpointHandler {
	return &PersistentCheckpointHandler{store: utilstore.NewMemStore()}
}

func TestPersistentCheckpointHandler(t *testing.T) {
	var err error
	handler := NewTestPersistentCheckpointHandler()
	port80 := int32(80)
	port443 := int32(443)
	proto := protocolTCP

	checkpoint1 := NewPodSandboxCheckpoint("ns1", "sandbox1")
	checkpoint1.Data.PortMappings = []*PortMapping{
		{
			&proto,
			&port80,
			&port80,
		},
		{
			&proto,
			&port443,
			&port443,
		},
	}
	checkpoint1.Data.HostNetwork = true

	checkpoints := []struct {
		podSandboxID      string
		checkpoint        *PodSandboxCheckpoint
		expectHostNetwork bool
	}{
		{
			"id1",
			checkpoint1,
			true,
		},
		{
			"id2",
			NewPodSandboxCheckpoint("ns2", "sandbox2"),
			false,
		},
	}

	for _, tc := range checkpoints {
		// Test CreateCheckpoints
		err = handler.CreateCheckpoint(tc.podSandboxID, tc.checkpoint)
		assert.NoError(t, err)

		// Test GetCheckpoints
		checkpoint, err := handler.GetCheckpoint(tc.podSandboxID)
		assert.NoError(t, err)
		assert.Equal(t, *checkpoint, *tc.checkpoint)
		assert.Equal(t, checkpoint.Data.HostNetwork, tc.expectHostNetwork)
	}
	// Test ListCheckpoints
	keys, err := handler.ListCheckpoints()
	assert.NoError(t, err)
	sort.Strings(keys)
	assert.Equal(t, keys, []string{"id1", "id2"})

	// Test RemoveCheckpoints
	err = handler.RemoveCheckpoint("id1")
	assert.NoError(t, err)
	// Test Remove Nonexisted Checkpoints
	err = handler.RemoveCheckpoint("id1")
	assert.NoError(t, err)

	// Test ListCheckpoints
	keys, err = handler.ListCheckpoints()
	assert.NoError(t, err)
	assert.Equal(t, keys, []string{"id2"})

	// Test Get NonExisted Checkpoint
	_, err = handler.GetCheckpoint("id1")
	assert.Error(t, err)
}
