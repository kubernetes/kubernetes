/*
Copyright 2020 The Kubernetes Authors.

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
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
)

const testingCheckpoint = "memorymanager_checkpoint_test"

// assertStateEqual marks provided test as failed if provided states differ
func assertStateEqual(t *testing.T, restoredState, expectedState State) {
	expectedMachineState := expectedState.GetMachineState()
	restoredMachineState := restoredState.GetMachineState()
	assert.Equal(t, expectedMachineState, restoredMachineState, "expected MachineState does not equal to restored one")

	expectedMemoryAssignments := expectedState.GetMemoryAssignments()
	restoredMemoryAssignments := restoredState.GetMemoryAssignments()
	assert.Equal(t, expectedMemoryAssignments, restoredMemoryAssignments, "state memory assignments mismatch")
}

func TestCheckpointStateRestore(t *testing.T) {
	testCases := []struct {
		description       string
		checkpointContent string
		expectedError     string
		expectedState     *stateMemory
	}{
		{
			"Restore non-existing checkpoint",
			"",
			"",
			&stateMemory{},
		},
		{
			"Restore valid checkpoint",
			`{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 4215593881
			}`,
			"",
			&stateMemory{
				assignments: ContainerMemoryAssignments{
					"pod": map[string][]Block{
						"container1": {
							{
								NUMAAffinity: []int{0},
								Type:         v1.ResourceMemory,
								Size:         512,
							},
						},
					},
				},
				machineState: NUMANodeMap{
					0: &NUMANodeState{
						MemoryMap: map[v1.ResourceName]*MemoryTable{
							v1.ResourceMemory: {
								Allocatable:    1536,
								Free:           1024,
								Reserved:       512,
								SystemReserved: 512,
								TotalMemSize:   2048,
							},
						},
					},
				},
			},
		},
		{
			"Restore checkpoint with invalid checksum",
			`{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"affinity":[0],"type":"memory","size":512}]}},
				"checksum": 101010
			}`,
			"checkpoint is corrupted",
			&stateMemory{},
		},
		{
			"Restore checkpoint with invalid JSON",
			`{`,
			"unexpected end of JSON input",
			&stateMemory{},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "memorymanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	// create checkpoint manager for testing
	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	assert.NoError(t, err, "could not create testing checkpoint manager")

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			assert.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

			// prepare checkpoint for testing
			if strings.TrimSpace(tc.checkpointContent) != "" {
				checkpoint := &testutil.MockCheckpoint{Content: tc.checkpointContent}
				assert.NoError(t, cpm.CreateCheckpoint(testingCheckpoint, checkpoint), "could not create testing checkpoint")
			}

			restoredState, err := NewCheckpointState(testingDir, testingCheckpoint, "static")
			if strings.TrimSpace(tc.expectedError) != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "could not restore state from checkpoint: "+tc.expectedError)
			} else {
				assert.NoError(t, err, "unexpected error while creating checkpointState")
				// compare state after restoration with the one expected
				assertStateEqual(t, restoredState, tc.expectedState)
			}
		})
	}
}

func TestCheckpointStateStore(t *testing.T) {
	expectedState := &stateMemory{
		assignments: ContainerMemoryAssignments{
			"pod": map[string][]Block{
				"container1": {
					{
						NUMAAffinity: []int{0},
						Type:         v1.ResourceMemory,
						Size:         1024,
					},
				},
			},
		},
		machineState: NUMANodeMap{
			0: &NUMANodeState{
				MemoryMap: map[v1.ResourceName]*MemoryTable{
					v1.ResourceMemory: {
						Allocatable:    1536,
						Free:           512,
						Reserved:       1024,
						SystemReserved: 512,
						TotalMemSize:   2048,
					},
				},
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "memorymanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	assert.NoError(t, err, "could not create testing checkpoint manager")

	assert.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

	cs1, err := NewCheckpointState(testingDir, testingCheckpoint, "static")
	assert.NoError(t, err, "could not create testing checkpointState instance")

	// set values of cs1 instance so they are stored in checkpoint and can be read by cs2
	cs1.SetMachineState(expectedState.machineState)
	cs1.SetMemoryAssignments(expectedState.assignments)

	// restore checkpoint with previously stored values
	cs2, err := NewCheckpointState(testingDir, testingCheckpoint, "static")
	assert.NoError(t, err, "could not create testing checkpointState instance")

	assertStateEqual(t, cs2, expectedState)
}

func TestCheckpointStateHelpers(t *testing.T) {
	testCases := []struct {
		description  string
		machineState NUMANodeMap
		assignments  ContainerMemoryAssignments
	}{
		{
			description: "One container",
			assignments: ContainerMemoryAssignments{
				"pod": map[string][]Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         1024,
						},
					},
				},
			},
			machineState: NUMANodeMap{
				0: &NUMANodeState{
					MemoryMap: map[v1.ResourceName]*MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536,
							Free:           512,
							Reserved:       1024,
							SystemReserved: 512,
							TotalMemSize:   2048,
						},
					},
					Cells: []int{},
				},
			},
		},
		{
			description: "Two containers",
			assignments: ContainerMemoryAssignments{
				"pod": map[string][]Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         512,
						},
					},
					"container2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         512,
						},
					},
				},
			},
			machineState: NUMANodeMap{
				0: &NUMANodeState{
					MemoryMap: map[v1.ResourceName]*MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536,
							Free:           512,
							Reserved:       1024,
							SystemReserved: 512,
							TotalMemSize:   2048,
						},
					},
					Cells: []int{},
				},
			},
		},
		{
			description: "Container without assigned memory",
			assignments: ContainerMemoryAssignments{
				"pod": map[string][]Block{
					"container1": {},
				},
			},
			machineState: NUMANodeMap{
				0: &NUMANodeState{
					MemoryMap: map[v1.ResourceName]*MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536,
							Free:           1536,
							Reserved:       0,
							SystemReserved: 512,
							TotalMemSize:   2048,
						},
					},
					Cells: []int{},
				},
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "memorymanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	assert.NoError(t, err, "could not create testing checkpoint manager")

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			assert.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

			state, err := NewCheckpointState(testingDir, testingCheckpoint, "static")
			assert.NoError(t, err, "could not create testing checkpoint manager")

			state.SetMachineState(tc.machineState)
			assert.Equal(t, tc.machineState, state.GetMachineState(), "machine state inconsistent")

			for pod := range tc.assignments {
				for container, blocks := range tc.assignments[pod] {
					state.SetMemoryBlocks(pod, container, blocks)
					assert.Equal(t, blocks, state.GetMemoryBlocks(pod, container), "memory block inconsistent")

					state.Delete(pod, container)
					assert.Nil(t, state.GetMemoryBlocks(pod, container), "deleted container still existing in state")
				}
			}
		})
	}
}

func TestCheckpointStateClear(t *testing.T) {
	testCases := []struct {
		description  string
		machineState NUMANodeMap
		assignments  ContainerMemoryAssignments
	}{
		{
			description: "Valid state cleaning",
			assignments: ContainerMemoryAssignments{
				"pod": map[string][]Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         1024,
						},
					},
				},
			},
			machineState: NUMANodeMap{
				0: &NUMANodeState{
					MemoryMap: map[v1.ResourceName]*MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536,
							Free:           512,
							Reserved:       1024,
							SystemReserved: 512,
							TotalMemSize:   2048,
						},
					},
				},
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "memorymanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			state, err := NewCheckpointState(testingDir, testingCheckpoint, "static")
			assert.NoError(t, err, "could not create testing checkpoint manager")

			state.SetMachineState(tc.machineState)
			state.SetMemoryAssignments(tc.assignments)

			state.ClearState()
			assert.Equal(t, NUMANodeMap{}, state.GetMachineState(), "cleared state with non-empty machine state")
			assert.Equal(t, ContainerMemoryAssignments{}, state.GetMemoryAssignments(), "cleared state with non-empty memory assignments")
		})
	}
}
