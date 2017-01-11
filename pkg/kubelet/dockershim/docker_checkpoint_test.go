/*
Copyright 2016 The Kubernetes Authors.

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
	"reflect"
	"sort"
	"testing"
)

func NewTestPersistentCheckpointHandler() CheckpointHandler {
	return &PersistentCheckpointHandler{store: NewMemStore()}
}

func TestPersistentCheckpointHandler(t *testing.T) {
	handler := NewTestPersistentCheckpointHandler()
	port80 := int32(80)
	port443 := int32(443)
	proto := ProtocolTCP

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

	checkpoints := []struct {
		podSandboxID string
		checkpoint   *PodSandboxCheckpoint
	}{
		{
			"id1",
			checkpoint1,
		},
		{
			"id2",
			NewPodSandboxCheckpoint("ns2", "sandbox2"),
		},
	}

	for _, tc := range checkpoints {
		// Test CreateCheckpoints
		err := handler.CreateCheckpoint(tc.podSandboxID, tc.checkpoint)
		if err != nil {
			t.Errorf("Do not expect error when creating checkpoint: %v", err)
		}
		// Test GetCheckpoints
		checkpoint, err := handler.GetCheckpoint(tc.podSandboxID)
		if err != nil {
			t.Errorf("Do not expect error when retriving checkpoint: %v", err)
		}
		if !reflect.DeepEqual(*checkpoint, *tc.checkpoint) {
			t.Errorf("Expect checkpoint data to be consistent")
		}
	}
	// Test ListCheckpoints
	keys := handler.ListCheckpoints()
	sort.Strings(keys)
	if !reflect.DeepEqual(keys, []string{"id1", "id2"}) {
		t.Errorf("ListCheckpoint does not return result as expected: %v", handler.ListCheckpoints())
	}

	// Test RemoveCheckpoints
	err := handler.RemoveCheckpoint("id1")
	if err != nil {
		t.Errorf("Failed to remove sandbox checkpoint")
	}
	// Test Remove Nonexisted Checkpoints
	err = handler.RemoveCheckpoint("id1")
	if err != nil {
		t.Errorf("Remove nonexisted sandbox checkpoint should not retrun error: %v", err)
	}

	// Test ListCheckpoints
	if !reflect.DeepEqual(handler.ListCheckpoints(), []string{"id2"}) {
		t.Errorf("ListCheckpoint does not return result as expected: %v", handler.ListCheckpoints())
	}

	// Test Get NonExisted Checkpoint
	_, err = handler.GetCheckpoint("id1")
	if err == nil {
		t.Errorf("Expect to get an error when retriving nonExisted checkpoint.")
	}
}
