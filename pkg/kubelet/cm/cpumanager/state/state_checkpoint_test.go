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
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/cpuset"
)

const testingCheckpoint = "cpumanager_checkpoint_test"

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
				defaultCPUSet: cpuset.New(4, 5, 6),
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
						"container1": cpuset.New(4, 5, 6),
						"container2": cpuset.New(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.New(1, 2, 3),
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
				defaultCPUSet: cpuset.New(1, 2, 3),
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
						"container1": cpuset.New(4, 5, 6),
						"container2": cpuset.New(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.New(1, 2, 3),
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "cpumanager_state_test")
	require.NoError(t, err)
	defer os.RemoveAll(testingDir)
	// create checkpoint manager for testing
	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoErrorf(t, err, "could not create testing checkpoint manager: %v", err)

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			cpm.RemoveCheckpoint(testingCheckpoint)

			// prepare checkpoint for testing
			if strings.TrimSpace(tc.checkpointContent) != "" {
				checkpoint := &testutil.MockCheckpoint{Content: tc.checkpointContent}
				err = cpm.CreateCheckpoint(testingCheckpoint, checkpoint)
				require.NoErrorf(t, err, "could not create testing checkpoint: %v", err)
			}

			logger, _ := ktesting.NewTestContext(t)
			restoredState, err := NewCheckpointState(logger, testingDir, testingCheckpoint, tc.policyName, tc.initialContainers)
			if strings.TrimSpace(tc.expectedError) == "" {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorContains(t, err, "could not restore state from checkpoint")
				require.ErrorContains(t, err, tc.expectedError)
				return
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
			&stateMemory{defaultCPUSet: cpuset.New(1, 2, 3)},
		},
		{
			"Store assignments",
			&stateMemory{
				assignments: map[string]map[string]cpuset.CPUSet{
					"pod": {
						"container1": cpuset.New(1, 5, 8),
					},
				},
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "cpumanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	if err != nil {
		t.Fatalf("could not create testing checkpoint manager: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			cpm.RemoveCheckpoint(testingCheckpoint)

			logger, _ := ktesting.NewTestContext(t)
			cs1, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}

			// set values of cs1 instance so they are stored in checkpoint and can be read by cs2
			cs1.SetDefaultCPUSet(tc.expectedState.defaultCPUSet)
			cs1.SetCPUAssignments(tc.expectedState.assignments)

			// restore checkpoint with previously stored values
			cs2, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
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
			defaultCPUset: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.New(0, 1),
				},
			},
		},
		{
			description:   "Two containers",
			defaultCPUset: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.New(0, 1),
					"c2": cpuset.New(2, 3, 4, 5),
				},
			},
		},
		{
			description:   "Container without assigned cpus",
			defaultCPUset: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.New(),
				},
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "cpumanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	if err != nil {
		t.Fatalf("could not create testing checkpoint manager: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			cpm.RemoveCheckpoint(testingCheckpoint)

			logger, _ := ktesting.NewTestContext(t)
			state, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
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
			cpuset.New(1, 5, 10),
			map[string]map[string]cpuset.CPUSet{
				"pod": {
					"container1": cpuset.New(1, 4),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// create temp dir
			testingDir, err := os.MkdirTemp("", "cpumanager_state_test")
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(testingDir)

			logger, _ := ktesting.NewTestContext(t)
			state, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}

			state.SetDefaultCPUSet(tc.defaultCPUset)
			state.SetCPUAssignments(tc.assignments)

			state.ClearState()
			if !cpuset.New().Equals(state.GetDefaultCPUSet()) {
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

func TestCheckpointStateHoldStore(t *testing.T) {
	podUID := "pod"
	initialState := &stateMemory{
		defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
		assignments: map[string]map[string]cpuset.CPUSet{
			podUID: {
				"c1": cpuset.New(0, 1),
				"c2": cpuset.New(2, 3, 4, 5),
				"c3": cpuset.New(),
			},
		},
	}
	testCases := []struct {
		description         string
		scenario            func(State)
		expectedCachedState *stateMemory
		expectedStoredState *stateMemory
	}{
		{
			"Hold storing cpu set",
			func(s State) {
				s.HoldStore()
				s.SetCPUSet(podUID, "c1", cpuset.New(6, 7))
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(6, 7),
						"c2": cpuset.New(2, 3, 4, 5),
						"c3": cpuset.New(),
					},
				},
			},
			initialState,
		},
		{
			"Hold storing default cpu set",
			func(s State) {
				s.HoldStore()
				s.SetDefaultCPUSet(cpuset.New(0, 1, 2, 3, 4, 5, 6, 9, 10))
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 9, 10),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(0, 1),
						"c2": cpuset.New(2, 3, 4, 5),
						"c3": cpuset.New(),
					},
				},
			},
			initialState,
		},
		{
			"Hold storing cpu to pod assignments",
			func(s State) {
				s.HoldStore()
				s.SetCPUAssignments(map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(),
						"c2": cpuset.New(2, 3),
						"c4": cpuset.New(0, 1),
					},
				})
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(),
						"c2": cpuset.New(2, 3),
						"c4": cpuset.New(0, 1),
					},
				},
			},
			initialState,
		},
		{
			"Hold storing deletion of cpu to pod assignments",
			func(s State) {
				s.HoldStore()
				s.Delete(podUID, "c2")
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(0, 1),
						"c3": cpuset.New(),
					},
				},
			},
			initialState,
		},
		{
			"Hold storing clearing the state",
			func(s State) {
				s.HoldStore()
				s.ClearState()
			},
			&stateMemory{},
			initialState,
		},
		{
			"Store cpu set after hold is disabled",
			func(s State) {
				s.HoldStore()
				defer s.Store()
				s.SetCPUSet(podUID, "c1", cpuset.New(6, 7))
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(6, 7),
						"c2": cpuset.New(2, 3, 4, 5),
						"c3": cpuset.New(),
					},
				},
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(6, 7),
						"c2": cpuset.New(2, 3, 4, 5),
						"c3": cpuset.New(),
					},
				},
			},
		},
		{
			"Store default cpu set after hold is disabled",
			func(s State) {
				s.HoldStore()
				defer s.Store()
				s.SetDefaultCPUSet(cpuset.New(0, 1, 2, 3, 4, 5, 6, 9, 10))
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 9, 10),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(0, 1),
						"c2": cpuset.New(2, 3, 4, 5),
						"c3": cpuset.New(),
					},
				},
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 9, 10),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(0, 1),
						"c2": cpuset.New(2, 3, 4, 5),
						"c3": cpuset.New(),
					},
				},
			},
		},
		{
			"Store cpu to pod assignments after hold is disabled",
			func(s State) {
				s.HoldStore()
				defer s.Store()
				s.SetCPUAssignments(map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(),
						"c2": cpuset.New(2, 3),
						"c4": cpuset.New(0, 1),
					},
				})
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(),
						"c2": cpuset.New(2, 3),
						"c4": cpuset.New(0, 1),
					},
				},
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(),
						"c2": cpuset.New(2, 3),
						"c4": cpuset.New(0, 1),
					},
				},
			},
		},
		{
			"Store deletion of cpu to pod assignments after hold is disabled",
			func(s State) {
				s.HoldStore()
				defer s.Store()
				s.Delete(podUID, "c2")
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(0, 1),
						"c3": cpuset.New(),
					},
				},
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(0, 1),
						"c3": cpuset.New(),
					},
				},
			},
		},
		{
			"Store clearing the state after hold is disabled",
			func(s State) {
				s.HoldStore()
				defer s.Store()
				s.ClearState()
			},
			&stateMemory{},
			&stateMemory{},
		},
		{
			"Continue regular storing after hold is disabled",
			func(s State) {
				s.HoldStore()
				s.SetCPUSet(podUID, "c1", cpuset.New(6, 7))
				s.Store()

				s.SetCPUSet(podUID, "c2", cpuset.New(8))
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(6, 7),
						"c2": cpuset.New(8),
						"c3": cpuset.New(),
					},
				},
			},
			&stateMemory{
				defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
				assignments: map[string]map[string]cpuset.CPUSet{
					podUID: {
						"c1": cpuset.New(6, 7),
						"c2": cpuset.New(8),
						"c3": cpuset.New(),
					},
				},
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "cpumanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	if err != nil {
		t.Fatalf("could not create testing checkpoint manager: %v", err)
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			cpm.RemoveCheckpoint(testingCheckpoint)

			logger, _ := ktesting.NewTestContext(t)
			cs1, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}

			// set initial values
			cs1.SetDefaultCPUSet(initialState.defaultCPUSet)
			cs1.SetCPUAssignments(initialState.assignments)

			// execute test case scenario
			tc.scenario(cs1)

			// verify cached state
			AssertStateEqual(t, cs1, tc.expectedCachedState)

			// restore checkpoint with previously stored values
			cs2, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
			if err != nil {
				t.Fatalf("could not create testing checkpointState instance: %v", err)
			}

			AssertStateEqual(t, cs2, tc.expectedStoredState)
		})
	}
}

func TestCheckpointStateHoldStoreWithNoChanges(t *testing.T) {
	podUID := "pod"
	initialState := &stateMemory{
		defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8),
		assignments: map[string]map[string]cpuset.CPUSet{
			podUID: {
				"c1": cpuset.New(0, 1),
				"c2": cpuset.New(2, 3, 4, 5),
				"c3": cpuset.New(),
			},
		},
	}
	modifiedState := &stateMemory{
		defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6),
		assignments: map[string]map[string]cpuset.CPUSet{
			"pod": {
				"c1": cpuset.New(0, 6),
				"c2": cpuset.New(),
				"c3": cpuset.New(3, 5),
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "cpumanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(testingDir)

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	if err != nil {
		t.Fatalf("could not create testing checkpoint manager: %v", err)
	}

	// ensure there is no previous checkpoint
	cpm.RemoveCheckpoint(testingCheckpoint)

	logger, _ := ktesting.NewTestContext(t)
	cs1, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
	if err != nil {
		t.Fatalf("could not create testing checkpointState instance: %v", err)
	}

	// set initial values
	cs1.SetDefaultCPUSet(initialState.defaultCPUSet)
	cs1.SetCPUAssignments(initialState.assignments)

	// create secondary checkpoint poining to same location
	cs2, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
	if err != nil {
		t.Fatalf("could not create testing checkpointState instance: %v", err)
	}

	// initiate hold in primary checkpoint
	cs1.HoldStore()

	// overwrite checkpoint file
	cs2.SetDefaultCPUSet(modifiedState.defaultCPUSet)
	cs2.SetCPUAssignments(modifiedState.assignments)

	// release hold on primary checkpoint
	cs1.Store()

	// verify the primary checkpoint cache is unchanged and contains initial values
	AssertStateEqual(t, cs1, initialState)

	// verify the stored values with tertiary checkpoint
	cs3, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
	if err != nil {
		t.Fatalf("could not create testing checkpointState instance: %v", err)
	}
	AssertStateEqual(t, cs3, modifiedState)
}

func AssertStateEqual(t *testing.T, sf State, sm State) {
	cpusetSf := sf.GetDefaultCPUSet()
	cpusetSm := sm.GetDefaultCPUSet()
	if !cpusetSf.Equals(cpusetSm) {
		t.Errorf("State CPUSet mismatch. Have %v, want %v", cpusetSf, cpusetSm)
	}

	cpuassignmentSf := sf.GetCPUAssignments()
	cpuassignmentSm := sm.GetCPUAssignments()
	if !reflect.DeepEqual(cpuassignmentSf, cpuassignmentSm) {
		t.Errorf("State CPU assignments mismatch. Have %s, want %s", cpuassignmentSf, cpuassignmentSm)
	}
}
