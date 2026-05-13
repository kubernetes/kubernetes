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
	"maps"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/cpuset"
)

const testingCheckpoint = "cpumanager_checkpoint_test"

type FeatureGateCombination map[featuregate.Feature]bool

// allFeatureGateCombinations generates all combinations of provided feature gates.
// Each combination is represented as a map of feature name to its state (enabled/disabled).
// Combinations are constructed:
//   - starting with single empty combination
//   - then for each feature gate all combinations are appended twice to new slice
//     (once with feature disabled and once with feature enabled)
//   - combination list is replaced with new slice
//
// For example, given two features A and B, allFeatureGateCombinations will build combinations:
// * Initial: `[{}]`
// * After A: `[{A: false}, {A: true}]`
// * After B: `[{A: false, B: false}, {A: false, B: true}, {A: true, B: false}, {A: true, B: true}]`
func allFeatureGateCombinations(gates []featuregate.Feature) []FeatureGateCombination {
	combinations := []FeatureGateCombination{make(FeatureGateCombination)}
	for _, gate := range gates {
		var newCombinations []FeatureGateCombination
		for _, combination := range combinations {
			// Append combination copy with the feature disabled
			disabled := maps.Clone(combination)
			disabled[gate] = false
			newCombinations = append(newCombinations, disabled)

			// Append combination with the feature enabled
			combination[gate] = true
			newCombinations = append(newCombinations, combination)
		}
		combinations = newCombinations
	}
	return combinations
}

func describe(comb FeatureGateCombination) string {
	keys := slices.Sorted(maps.Keys(comb))
	if len(keys) == 0 {
		return ""
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "%s=%v", keys[0], comb[keys[0]])
	for _, key := range keys[1:] {
		fmt.Fprintf(&sb, ",%s=%v", key, comb[key])
	}
	return sb.String()
}

func TestCheckpointStateRestore(t *testing.T) {
	testCases := []struct {
		description       string
		fgRequirements    FeatureGateCombination
		checkpointContent string
		policyName        string
		initialContainers containermap.ContainerMap
		expectedError     string
		expectedState     *stateMemory
	}{
		{
			"Restore non-existing checkpoint",
			nil,
			"",
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{},
		},
		{
			"Fail to restore checkpoint without data section",
			nil,
			`{
				"checksum": 1234
			}`,
			"none",
			containermap.ContainerMap{},
			"checkpoint is corrupted",
			nil,
		},
		{
			"Fail to restore checkpoint without checksum section (fall back to empty v2 version)",
			nil,
			`{
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"4-6\"}"
			}`,
			"none",
			containermap.ContainerMap{},
			`configured policy "none" differs from state checkpoint policy ""`,
			nil,
		},
		{
			"Restore default cpu set",
			nil,
			`{
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"4-6\"}",
				"checksum": 657950972
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
			nil,
			`{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"7-9\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"1-3\"}}}",
				"checksum": 1420829534
			}`,
			"static",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.New(4, 5, 6),
						"container2": cpuset.New(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.New(7, 8, 9),
			},
		},
		{
			"Fail to restore checkpoint with invalid checksum",
			nil,
			`{
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"4-6\"}",
				"checksum": 1234
			}`,
			"none",
			containermap.ContainerMap{},
			"checkpoint is corrupted",
			nil,
		},
		{
			"Fail to restore checkpoint with invalid JSON",
			nil,
			`{`,
			"none",
			containermap.ContainerMap{},
			"unexpected end of JSON input",
			nil,
		},
		{
			"Fail to restore checkpoint with invalid policy name",
			nil,
			`{
				"data": "{\"policyName\":\"other\",\"defaultCPUSet\":\"1-3\"}",
				"checksum": 2380595610
			}`,
			"none",
			containermap.ContainerMap{},
			`configured policy "none" differs from state checkpoint policy "other"`,
			nil,
		},
		{
			"Fail to restore checkpoint with unparsable default cpu set",
			nil,
			`{
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"1.3\"}",
				"checksum": 3033143655
			}`,
			"none",
			containermap.ContainerMap{},
			`could not parse default cpu set "1.3": strconv.Atoi: parsing "1.3": invalid syntax`,
			nil,
		},
		{
			"Fail to restore checkpoint with unparsable assignment entry",
			nil,
			`{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"asd\"}}}",
				"checksum": 3794806925
			}`,
			"static",
			containermap.ContainerMap{},
			`could not parse cpuset "asd" for container "container2" in pod "pod": strconv.Atoi: parsing "asd": invalid syntax`,
			nil,
		},
		{
			"Restore checkpoint ignoring unknown fields in data section",
			nil,
			`{
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"4-6\",\"unknownField\":\"value\"}",
				"checksum": 3492408555
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				defaultCPUSet: cpuset.New(4, 5, 6),
			},
		},
		{
			"Restore checkpoint from checkpoint with v1 checksum",
			nil,
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
			"Restore checkpoint from v1 (migration)",
			nil,
			`{
				"policyName": "static",
				"defaultCPUSet": "7-9",
				"entries": {
					"containerID1": "4-6",
					"containerID2": "1-3"
				},
				"checksum": 2026311253
			}`,
			"static",
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
				defaultCPUSet: cpuset.New(7, 8, 9),
			},
		},
		{
			"Restore checkpoint from v2 (migration)",
			nil,
			`{
				"policyName": "static",
				"defaultCPUSet": "7-9",
				"entries": {
					"pod": {
						"container1": "4-6",
						"container2": "1-3"
					}
				},
				"checksum": 1942532442
			}`,
			"static",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.New(4, 5, 6),
						"container2": cpuset.New(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.New(7, 8, 9),
			},
		},
		{
			"Restore checkpoint from v3 (migration) with PodLevelResourceManagers disabled",
			FeatureGateCombination{features.PodLevelResourceManagers: false},
			`{
				"policyName": "static",
				"defaultCPUSet": "1-2",
				"entries": {
					"pod1": {
						"container1": "5-6",
						"container2": "3-4"
					}
				},
				"podEntries": {
					"pod2": {
						"cpuSet":"7-9"
					}
				},
				"checksum": 2284712151
			}`,
			"static",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod1": map[string]cpuset.CPUSet{
						"container1": cpuset.New(5, 6),
						"container2": cpuset.New(3, 4),
					},
				},
				defaultCPUSet: cpuset.New(1, 2),
			},
		},
		{
			"Restore checkpoint from v3 (migration) with PodLevelResourceManagers enabled",
			FeatureGateCombination{features.PodLevelResourceManagers: true},
			`{
				"policyName": "static",
				"defaultCPUSet": "1-2",
				"entries": {
					"pod1": {
						"container1": "5-6",
						"container2": "3-4"
					}
				},
				"podEntries": {
					"pod2": {
						"cpuSet":"7-9"
					}
				},
				"checksum": 2284712151
			}`,
			"static",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod1": map[string]cpuset.CPUSet{
						"container1": cpuset.New(5, 6),
						"container2": cpuset.New(3, 4),
					},
				},
				defaultCPUSet: cpuset.New(1, 2),
				podAssignments: PodCPUAssignments{
					"pod2": PodEntry{
						CPUSet: cpuset.New(7, 8, 9),
					},
				},
			},
		},
		{
			"Restore valid v4 checkpoint with PodLevelResourceManagers disabled",
			FeatureGateCombination{features.PodLevelResourceManagers: false},
			`{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-10\"}}}",
				"checksum": 2328898362
			}`,
			"static",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.New(4, 5, 6),
						"container2": cpuset.New(7, 8, 9),
					},
				},
				defaultCPUSet: cpuset.New(1, 2, 3),
			},
		},
		{
			"Restore valid v4 checkpoint with PodLevelResourceManagers enabled",
			FeatureGateCombination{features.PodLevelResourceManagers: true},
			`{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-10\"}}}",
				"checksum": 2328898362
			}`,
			"static",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.New(4, 5, 6),
						"container2": cpuset.New(7, 8, 9),
					},
				},
				defaultCPUSet: cpuset.New(1, 2, 3),
				podAssignments: PodCPUAssignments{
					"pod": PodEntry{
						CPUSet: cpuset.New(4, 5, 6, 7, 8, 9, 10),
					},
				},
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "cpumanager_state_test")
	require.NoError(t, err)
	defer removeAll(testingDir, t)
	// create checkpoint manager for testing
	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoErrorf(t, err, "could not create testing checkpoint manager: %v", err)

	// list of all features verified in this test
	featureGateList := []featuregate.Feature{
		features.PodLevelResourceManagers,
	}
	// iterate over all possible enabled/disabled feature combinations
	for _, fgComb := range allFeatureGateCombinations(featureGateList) {
		// run all testcases for current feature combination
		t.Run(describe(fgComb), func(t *testing.T) {
			for _, key := range slices.Sorted(maps.Keys(fgComb)) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, key, fgComb[key])
			}

			for _, tc := range testCases {
				// verify feature gate requirements for testcase
				skip := false
				for fg, requiredState := range tc.fgRequirements {
					state, exist := fgComb[fg]
					if !exist || requiredState != state {
						skip = true
					}
				}
				if skip {
					continue
				}

				t.Run(tc.description, func(t *testing.T) {
					// ensure there is no previous checkpoint
					err = cpm.RemoveCheckpoint(testingCheckpoint)
					require.NoErrorf(t, err, "could not remove previous checkpoint: %v", err)

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

					AssertStateEqual(t, restoredState, tc.expectedState)
				})
			}
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
	defer removeAll(testingDir, t)

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
	defer removeAll(testingDir, t)

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
			defer removeAll(testingDir, t)

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

	podcpuassignmentSf := sf.GetPodCPUAssignments()
	podcpuassignmentSm := sm.GetPodCPUAssignments()
	if !reflect.DeepEqual(podcpuassignmentSf, podcpuassignmentSm) {
		t.Errorf("State CPU assignments mismatch. Have %s, want %s", podcpuassignmentSf, podcpuassignmentSm)
	}
}

func TestCPUManagerCheckpointV2_MarshalCheckpoint_ForwardCompatibility(t *testing.T) {
	// 1. Create a V2 checkpoint using the struct defined in the current codebase (1.36+)
	currentCheckpoint := &CPUManagerCheckpointV2{
		PolicyName:    "none",
		DefaultCPUSet: "1-3",
		Entries:       make(map[string]map[string]string),
	}

	// Marshal it using the logic that forces the "CPUManagerCheckpoint" name
	data, err := currentCheckpoint.MarshalCheckpoint()
	if err != nil {
		t.Fatalf("Failed to marshal checkpoint: %v", err)
	}

	// 2. Unmarshal the raw JSON to extract the checksum that was actually written to the file
	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("Failed to unmarshal JSON: %v", err)
	}

	actualChecksumFloat, ok := result["checksum"].(float64)
	if !ok {
		t.Fatalf("Checksum field missing or invalid type")
	}
	writtenChecksum := checksum.Checksum(uint64(actualChecksumFloat))

	// 3. Reconstruct how versions 1.35 and earlier would calculate the checksum
	// by defining a struct with the exact legacy name and fields.
	type CPUManagerCheckpoint struct {
		PolicyName    string                       `json:"policyName"`
		DefaultCPUSet string                       `json:"defaultCpuSet"`
		Entries       map[string]map[string]string `json:"entries,omitempty"`
		Checksum      checksum.Checksum            `json:"checksum"`
	}

	legacyCheckpoint := &CPUManagerCheckpoint{
		PolicyName:    currentCheckpoint.PolicyName,
		DefaultCPUSet: currentCheckpoint.DefaultCPUSet,
		Entries:       currentCheckpoint.Entries,
	}

	expectedLegacyChecksum := checksum.New(legacyCheckpoint)

	// 4. Assert that the checksum written by our 1.36+ code matches
	// what a 1.35 Kubelet would expect to see.
	if writtenChecksum != expectedLegacyChecksum {
		t.Errorf("Written Checksum %d does not match legacy calculation %d. Forward compatibility broken.", writtenChecksum, expectedLegacyChecksum)
	}
}

func removeAll(dir string, t *testing.T) {
	t.Helper()
	if err := os.RemoveAll(dir); err != nil {
		t.Fatalf("unable to remove dir %s: %v", dir, err)
	}
}
