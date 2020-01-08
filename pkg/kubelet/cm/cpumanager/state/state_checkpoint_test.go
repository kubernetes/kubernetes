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
	"os"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/containermap"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

const testingCheckpoint = "cpumanager_checkpoint_test"

var testingDir = os.TempDir()

func TestCheckpointStateRestore(t *testing.T) {
	testCases := []struct {
		description       string
		checkpointContent string
		policyName        string
		initialContainers containermap.ContainerMap
		expectedError     string
		expectedState     *stateMemory
	}{
		{
			"Restore non-existing checkpoint",
			"",
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{},
		},
		{
			"Restore default cpu set",
			`{
				"policyName": "none",
				"defaultCPUSet": "4-6",
				"entries": {},
				"checksum": 354655845
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				defaultCPUSet: cpuset.NewCPUSet(4, 5, 6),
			},
		},
		{
			"Restore valid checkpoint",
			`{
				"policyName": "none",
				"defaultCPUSet": "1-3",
				"entries": {
					"pod": {
						"container1": "4-6",
						"container2": "1-3"
					}
				},
				"checksum": 3610638499
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.NewCPUSet(4, 5, 6),
						"container2": cpuset.NewCPUSet(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.NewCPUSet(1, 2, 3),
			},
		},
		{
			"Restore checkpoint with invalid checksum",
			`{
				"policyName": "none",
				"defaultCPUSet": "4-6",
				"entries": {},
				"checksum": 1337
			}`,
			"none",
			containermap.ContainerMap{},
			"checkpoint is corrupted",
			&stateMemory{},
		},
		{
			"Restore checkpoint with invalid JSON",
			`{`,
			"none",
			containermap.ContainerMap{},
			"unexpected end of JSON input",
			&stateMemory{},
		},
		{
			"Restore checkpoint with invalid policy name",
			`{
				"policyName": "other",
				"defaultCPUSet": "1-3",
				"entries": {},
				"checksum": 1394507217
			}`,
			"none",
			containermap.ContainerMap{},
			`configured policy "none" differs from state checkpoint policy "other"`,
			&stateMemory{},
		},
		{
			"Restore checkpoint with unparsable default cpu set",
			`{
				"policyName": "none",
				"defaultCPUSet": "1.3",
				"entries": {},
				"checksum": 3021697696
			}`,
			"none",
			containermap.ContainerMap{},
			`could not parse default cpu set "1.3": strconv.Atoi: parsing "1.3": invalid syntax`,
			&stateMemory{},
		},
		{
			"Restore checkpoint with unparsable assignment entry",
			`{
				"policyName": "none",
				"defaultCPUSet": "1-3",
				"entries": {
					"pod": {
						"container1": "4-6",
						"container2": "asd"
					}
				},
				"checksum": 962272150
			}`,
			"none",
			containermap.ContainerMap{},
			`could not parse cpuset "asd" for container "container2" in pod "pod": strconv.Atoi: parsing "asd": invalid syntax`,
			&stateMemory{},
		},
		{
			"Restore checkpoint from checkpoint with v1 checksum",
			`{
				"policyName": "none",
				"defaultCPUSet": "1-3",
				"checksum": 1694838852
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				defaultCPUSet: cpuset.NewCPUSet(1, 2, 3),
			},
		},
		{
			"Restore checkpoint with migration",
			`{
				"policyName": "none",
				"defaultCPUSet": "1-3",
				"entries": {
					"containerID1": "4-6",
					"containerID2": "1-3"
				},
				"checksum": 3680390589
			}`,
			"none",
			func() containermap.ContainerMap {
				cm := containermap.NewContainerMap()
				cm.Add("pod", "container1", "containerID1")
				cm.Add("pod", "container2", "containerID2")
				return cm
			}(),
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.NewCPUSet(4, 5, 6),
						"container2": cpuset.NewCPUSet(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.NewCPUSet(1, 2, 3),
			},
		},
	}

	// create checkpoint manager for testing
	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	if err != nil {
		t.Fatalf("could not create testing checkpoint manager: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			cpm.RemoveCheckpoint(testingCheckpoint)

			// prepare checkpoint for testing
			if strings.TrimSpace(tc.checkpointContent) != "" {
				checkpoint := &testutil.MockCheckpoint{Content: tc.checkpointContent}
				if err := cpm.CreateCheckpoint(testingCheckpoint, checkpoint); err != nil {
					t.Fatalf("could not create testing checkpoint: %v", err)
				}
			}

			restoredState, err := NewCheckpointState(testingDir, testingCheckpoint, tc.policyName, tc.initialContainers)
			if err != nil {
				if strings.TrimSpace(tc.expectedError) != "" {
					tc.expectedError = "could not restore state from checkpoint: " + tc.expectedError
					if strings.HasPrefix(err.Error(), tc.expectedError) {
						t.Logf("got expected error: %v", err)
						return
					}
				}
				t.Fatalf("unexpected error while creatng checkpointState: %v", err)
			}

			// compare state after restoration with the one expected
			AssertStateEqual(t, restoredState, tc.expectedState)
		})
	}
}

func TestCheckpointStateStore(t *testing.T) {
	testCases := []struct {
		description   string
		expectedState *stateMemory
	}{
		{
			"Store default cpu set",
			&stateMemory{defaultCPUSet: cpuset.NewCPUSet(1, 2, 3)},
		},
		{
			"Store assignments",
			&stateMemory{
				assignments: map[string]map[string]cpuset.CPUSet{
					"pod": {
						"container1": cpuset.NewCPUSet(1, 5, 8),
					},
				},
			},
		},
	}

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	if err != nil {
		t.Fatalf("could not create testing checkpoint manager: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			cpm.RemoveCheckpoint(testingCheckpoint)

			cs1, err := NewCheckpointState(testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}

			// set values of cs1 instance so they are stored in checkpoint and can be read by cs2
			cs1.SetDefaultCPUSet(tc.expectedState.defaultCPUSet)
			cs1.SetCPUAssignments(tc.expectedState.assignments)

			// restore checkpoint with previously stored values
			cs2, err := NewCheckpointState(testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}

			AssertStateEqual(t, cs2, tc.expectedState)
		})
	}
}

func TestCheckpointStateHelpers(t *testing.T) {
	testCases := []struct {
		description   string
		defaultCPUset cpuset.CPUSet
		assignments   map[string]map[string]cpuset.CPUSet
	}{
		{
			description:   "One container",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(0, 1),
				},
			},
		},
		{
			description:   "Two containers",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(0, 1),
					"c2": cpuset.NewCPUSet(2, 3, 4, 5),
				},
			},
		},
		{
			description:   "Container without assigned cpus",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(),
				},
			},
		},
	}

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	if err != nil {
		t.Fatalf("could not create testing checkpoint manager: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			cpm.RemoveCheckpoint(testingCheckpoint)

			state, err := NewCheckpointState(testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}
			state.SetDefaultCPUSet(tc.defaultCPUset)

			for pod := range tc.assignments {
				for container, set := range tc.assignments[pod] {
					state.SetCPUSet(pod, container, set)
					if cpus, _ := state.GetCPUSet(pod, container); !cpus.Equals(set) {
						t.Fatalf("state inconsistent, got %q instead of %q", set, cpus)
					}

					state.Delete(pod, container)
					if _, ok := state.GetCPUSet(pod, container); ok {
						t.Fatal("deleted container still existing in state")
					}
				}
			}
		})
	}
}

func TestCheckpointStateClear(t *testing.T) {
	testCases := []struct {
		description   string
		defaultCPUset cpuset.CPUSet
		assignments   map[string]map[string]cpuset.CPUSet
	}{
		{
			"Valid state",
			cpuset.NewCPUSet(1, 5, 10),
			map[string]map[string]cpuset.CPUSet{
				"pod": {
					"container1": cpuset.NewCPUSet(1, 4),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			state, err := NewCheckpointState(testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}

			state.SetDefaultCPUSet(tc.defaultCPUset)
			state.SetCPUAssignments(tc.assignments)

			state.ClearState()
			if !cpuset.NewCPUSet().Equals(state.GetDefaultCPUSet()) {
				t.Fatal("cleared state with non-empty default cpu set")
			}
			for pod := range tc.assignments {
				for container := range tc.assignments[pod] {
					if _, ok := state.GetCPUSet(pod, container); ok {
						t.Fatalf("container %q in pod %q with non-default cpu set in cleared state", container, pod)
					}
				}
			}
		})
	}
}
