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
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const testingCheckpoint = "memorymanager_checkpoint_test"

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

// assertStateEqual marks provided test as failed if provided states differ
func assertStateEqual(t *testing.T, restoredState, expectedState State) {
	expectedMachineState := expectedState.GetMachineState()
	restoredMachineState := restoredState.GetMachineState()
	require.Equal(t, expectedMachineState, restoredMachineState, "expected MachineState does not equal to restored one")

	expectedMemoryAssignments := expectedState.GetMemoryAssignments()
	restoredMemoryAssignments := restoredState.GetMemoryAssignments()
	require.Equal(t, expectedMemoryAssignments, restoredMemoryAssignments, "state memory assignments mismatch")

	expectedPodMemoryAssignments := expectedState.GetPodMemoryAssignments()
	restoredPodMemoryAssignments := restoredState.GetPodMemoryAssignments()
	require.Equal(t, expectedPodMemoryAssignments, restoredPodMemoryAssignments, "state memory pod assignments mismatch")
}

func TestCheckpointStateRestore(t *testing.T) {
	testCases := []struct {
		description       string
		fgRequirements    FeatureGateCombination
		checkpointContent string
		policyName        string
		expectedError     string
		expectedState     *stateMemory
	}{
		{
			description:       "Restore non-existing checkpoint",
			fgRequirements:    nil,
			checkpointContent: "",
			policyName:        "none",
			expectedError:     "",
			expectedState:     &stateMemory{},
		},
		{
			description:    "Fail to restore checkpoint without data section",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 4215593881,
				"dataChecksum": 1234
			}`,
			policyName:    "none",
			expectedError: "could not load v3 checkpoint: failed to deserialize memory manager checkpoint data: unexpected end of JSON input",
			expectedState: &stateMemory{},
		},
		{
			description:    "Fail to restore checkpoint without dataChecksum section",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 4215593881,
				"data": "{\"policyName\":\"static\",\"machineState\":{\"0\":{\"numberOfAssignments\":0,\"memoryMap\":{\"memory\":{\"total\":2048,\"systemReserved\":512,\"allocatable\":1536,\"reserved\":512,\"free\":1024}},\"cells\":[]}},\"entries\":{\"pod\":{\"container1\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}}}"
			}`,
			policyName:    "none",
			expectedError: "could not load v3 checkpoint: checkpoint is corrupted",
			expectedState: &stateMemory{},
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			description:    "Restore valid checkpoint",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"other",
				"machineState":{"0":{"numberOfAssignments":5,"memoryMap":{"memory":{"total":1024,"systemReserved":256,"allocatable":768,"reserved":256,"free":512}},"cells":[]}},
				"entries":{"pod":{"containerX":[{"numaAffinity":[0],"type":"memory","size":128}]}},
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"machineState\":{\"0\":{\"numberOfAssignments\":0,\"memoryMap\":{\"memory\":{\"total\":2048,\"systemReserved\":512,\"allocatable\":1536,\"reserved\":512,\"free\":1024}},\"cells\":[]}},\"entries\":{\"pod\":{\"container1\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}}}",
				"dataChecksum": 1849615440
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
			description:    "Fail to restore checkpoint with invalid dataChecksum",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 4215593881,
				"data": "{\"policyName\":\"static\",\"machineState\":{\"0\":{\"numberOfAssignments\":0,\"memoryMap\":{\"memory\":{\"total\":2048,\"systemReserved\":512,\"allocatable\":1536,\"reserved\":512,\"free\":1024}},\"cells\":[]}},\"entries\":{\"pod\":{\"container1\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}}}",
				"dataChecksum": 1234
			}`,
			policyName:    "none",
			expectedError: "could not load v3 checkpoint: checkpoint is corrupted",
			expectedState: &stateMemory{},
		},
		{
			description:    "Fail to restore checkpoint with invalid data JSON",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 4215593881,
				"data": "{",
				"dataChecksum": 1234
			}`,
			policyName:    "none",
			expectedError: "could not load v3 checkpoint: failed to deserialize memory manager checkpoint data: unexpected end of JSON input",
			expectedState: &stateMemory{},
		},
		{
			description:       "Fail to restore checkpoint with invalid JSON",
			fgRequirements:    nil,
			checkpointContent: `{`,
			policyName:        "none",
			expectedError:     "unexpected end of JSON input",
			expectedState:     &stateMemory{},
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			description:    "Fail to restore checkpoint with invalid policy name",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"other",
				"machineState":{"0":{"numberOfAssignments":5,"memoryMap":{"memory":{"total":1024,"systemReserved":256,"allocatable":768,"reserved":256,"free":512}},"cells":[]}},
				"entries":{"pod":{"containerX":[{"numaAffinity":[0],"type":"memory","size":128}]}},
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"machineState\":{\"0\":{\"numberOfAssignments\":0,\"memoryMap\":{\"memory\":{\"total\":2048,\"systemReserved\":512,\"allocatable\":1536,\"reserved\":512,\"free\":1024}},\"cells\":[]}},\"entries\":{\"pod\":{\"container1\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}}}",
				"dataChecksum": 1849615440
			}`,
			policyName:    "none",
			expectedError: `[memorymanager] configured policy "none" differs from state checkpoint policy "static"`,
			expectedState: &stateMemory{},
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			description:    "Restore checkpoint ignoring unknown fields in data section",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"other",
				"machineState":{"0":{"numberOfAssignments":5,"memoryMap":{"memory":{"total":1024,"systemReserved":256,"allocatable":768,"reserved":256,"free":512}},"cells":[]}},
				"entries":{"pod":{"containerX":[{"numaAffinity":[0],"type":"memory","size":128}]}},
				"checksum": 1234,
				"data": "{\"unknownField\":\"value\",\"policyName\":\"static\",\"machineState\":{\"0\":{\"numberOfAssignments\":0,\"memoryMap\":{\"memory\":{\"total\":2048,\"systemReserved\":512,\"allocatable\":1536,\"reserved\":512,\"free\":1024}},\"cells\":[]}},\"entries\":{\"pod\":{\"container1\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}}}",
				"dataChecksum": 2100279793
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
			description:    "Restore checkpoint from v1 (migration)",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 4215593881
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
			description:    "Restore checkpoint from v1 (migration) with checksum zeroed",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 0
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
			description:    "Fail to restore checkpoint from v1 with invalid checksum",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"affinity":[0],"type":"memory","size":512}]}},
				"checksum": 101010
			}`,
			policyName:    "static",
			expectedError: "checkpoint is corrupted",
			expectedState: &stateMemory{},
		},
		{
			description:    "Restore checkpoint from v2 with PodLevelResourceManagers enabled",
			fgRequirements: FeatureGateCombination{features.PodLevelResourceManagers: true},
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"podEntries":{"pod":{"memoryBlocks":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 1278640530
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
				podAssignments: PodMemoryAssignments{
					"pod": PodEntry{
						MemoryBlocks: []Block{
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
			description:    "Restore checkpoint from v2 with PodLevelResourceManagers disabled",
			fgRequirements: FeatureGateCombination{features.PodLevelResourceManagers: false},
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"podEntries":{"pod":{"memoryBlocks":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 1278640530
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
			description:    "Restore checkpoint from v2 with zeroed checksum and with PodLevelResourceManagers disabled",
			fgRequirements: FeatureGateCombination{features.PodLevelResourceManagers: false},
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"podEntries":{"pod":{"memoryBlocks":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 0
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
			description:    "Fail to restore checkpoint from v2 with invalid checksum",
			fgRequirements: nil,
			checkpointContent: `{
				"policyName":"static",
				"machineState":{"0":{"numberOfAssignments":0,"memoryMap":{"memory":{"total":2048,"systemReserved":512,"allocatable":1536,"reserved":512,"free":1024}},"cells":[]}},
				"entries":{"pod":{"container1":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"podEntries":{"pod":{"memoryBlocks":[{"numaAffinity":[0],"type":"memory","size":512}]}},
				"checksum": 101010
			}`,
			policyName:    "static",
			expectedError: "checkpoint is corrupted",
			expectedState: &stateMemory{},
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			description:    "Restore checkpoint from v3 with PodLevelResourceManagers enabled",
			fgRequirements: FeatureGateCombination{features.PodLevelResourceManagers: true},
			checkpointContent: `{
				"policyName":"other",
				"machineState":{"0":{"numberOfAssignments":5,"memoryMap":{"memory":{"total":1024,"systemReserved":256,"allocatable":768,"reserved":256,"free":512}},"cells":[]}},
				"entries":{"pod":{"containerX":[{"numaAffinity":[0],"type":"memory","size":128}]}},
				"podEntries":{"podY":{"memoryBlocks":[{"numaAffinity":[0],"type":"memory","size":64}]}},
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"machineState\":{\"0\":{\"numberOfAssignments\":0,\"memoryMap\":{\"memory\":{\"total\":2048,\"systemReserved\":512,\"allocatable\":1536,\"reserved\":512,\"free\":1024}},\"cells\":[]}},\"entries\":{\"pod\":{\"container1\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}},\"podEntries\":{\"pod\":{\"memoryBlocks\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}}}",
				"dataChecksum": 637296054
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
				podAssignments: PodMemoryAssignments{
					"pod": PodEntry{
						MemoryBlocks: []Block{
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
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			description:    "Restore checkpoint from v3 with PodLevelResourceManagers disabled",
			fgRequirements: FeatureGateCombination{features.PodLevelResourceManagers: false},
			checkpointContent: `{
				"policyName":"other",
				"machineState":{"0":{"numberOfAssignments":5,"memoryMap":{"memory":{"total":1024,"systemReserved":256,"allocatable":768,"reserved":256,"free":512}},"cells":[]}},
				"entries":{"pod":{"containerX":[{"numaAffinity":[0],"type":"memory","size":128}]}},
				"podEntries":{"podY":{"memoryBlocks":[{"numaAffinity":[0],"type":"memory","size":64}]}},
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"machineState\":{\"0\":{\"numberOfAssignments\":0,\"memoryMap\":{\"memory\":{\"total\":2048,\"systemReserved\":512,\"allocatable\":1536,\"reserved\":512,\"free\":1024}},\"cells\":[]}},\"entries\":{\"pod\":{\"container1\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}},\"podEntries\":{\"pod\":{\"memoryBlocks\":[{\"numaAffinity\":[0],\"type\":\"memory\",\"size\":512}]}}}",
				"dataChecksum": 637296054
			}`,
			policyName:    "static",
			expectedError: "",
			expectedState: &stateMemory{
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
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "memorymanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(testingDir), "unable to remove dir %s", testingDir)
	})

	// create checkpoint manager for testing
	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "could not create testing checkpoint manager")

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
					require.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

					// prepare checkpoint for testing
					if strings.TrimSpace(tc.checkpointContent) != "" {
						checkpoint := &testutil.MockCheckpoint{Content: tc.checkpointContent}
						require.NoError(t, cpm.CreateCheckpoint(testingCheckpoint, checkpoint), "could not create testing checkpoint")
					}

					logger, _ := ktesting.NewTestContext(t)
					restoredState, err := NewCheckpointState(logger, testingDir, testingCheckpoint, tc.policyName)
					if strings.TrimSpace(tc.expectedError) != "" {
						require.Error(t, err)
						require.ErrorContains(t, err, "could not restore state from checkpoint: "+tc.expectedError)
					} else {
						require.NoError(t, err, "unexpected error while creating checkpointState")
						// compare state after restoration with the one expected
						assertStateEqual(t, restoredState, tc.expectedState)
					}
				})
			}
		})
	}
}

func TestCheckpointStateStore(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
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
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(testingDir), "unable to remove dir %s", testingDir)
	})

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "could not create testing checkpoint manager")

	require.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

	cs1, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "static")
	require.NoError(t, err, "could not create testing checkpointState instance")

	// set values of cs1 instance so they are stored in checkpoint and can be read by cs2
	cs1.SetMachineState(expectedState.machineState)
	cs1.SetMemoryAssignments(expectedState.assignments)

	// restore checkpoint with previously stored values
	cs2, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "static")
	require.NoError(t, err, "could not create testing checkpointState instance")

	assertStateEqual(t, cs2, expectedState)
}

func TestCheckpointStateStore_WithPodLevelResourceManagersEnabled(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, true)
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
		podAssignments: PodMemoryAssignments{
			"pod": PodEntry{
				MemoryBlocks: []Block{
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
			},
		},
	}

	// create temp dir
	testingDir, err := os.MkdirTemp("", "memorymanager_state_test")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(testingDir), "unable to remove dir %s", testingDir)
	})

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "could not create testing checkpoint manager")

	require.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

	cs1, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "static")
	require.NoError(t, err, "could not create testing checkpointState instance")

	// set values of cs1 instance so they are stored in checkpoint and can be read by cs2
	cs1.SetMachineState(expectedState.machineState)
	cs1.SetMemoryAssignments(expectedState.assignments)
	cs1.SetPodMemoryAssignments(expectedState.podAssignments)

	// restore checkpoint with previously stored values
	cs2, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "static")
	require.NoError(t, err, "could not create testing checkpointState instance")

	assertStateEqual(t, cs2, expectedState)
}

func TestCheckpointStateHelpers(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
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
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(testingDir), "unable to remove dir %s", testingDir)
	})

	cpm, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "could not create testing checkpoint manager")

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// ensure there is no previous checkpoint
			require.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove testing checkpoint")

			state, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "static")
			require.NoError(t, err, "could not create testing checkpoint manager")

			state.SetMachineState(tc.machineState)
			require.Equal(t, tc.machineState, state.GetMachineState(), "machine state inconsistent")

			for pod := range tc.assignments {
				for container, blocks := range tc.assignments[pod] {
					state.SetMemoryBlocks(pod, container, blocks)
					require.Equal(t, blocks, state.GetMemoryBlocks(pod, container), "memory block inconsistent")

					state.Delete(pod, container)
					require.Nil(t, state.GetMemoryBlocks(pod, container), "deleted container still existing in state")
				}
			}
		})
	}
}

func TestCheckpointStateClear(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
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
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(testingDir), "unable to remove dir %s", testingDir)
	})

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			state, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "static")
			require.NoError(t, err, "could not create testing checkpoint manager")

			state.SetMachineState(tc.machineState)
			state.SetMemoryAssignments(tc.assignments)

			state.ClearState()
			require.Equal(t, NUMANodeMap{}, state.GetMachineState(), "cleared state with non-empty machine state")
			require.Equal(t, ContainerMemoryAssignments{}, state.GetMemoryAssignments(), "cleared state with non-empty memory assignments")
		})
	}
}

func TestMemoryManagerCheckpoint_MarshalCheckpoint_HashCompatibility(t *testing.T) {
	testCases := []struct {
		name                   string
		currentCheckpoint      any
		expectedLegacyChecksum func() checksum.Checksum
	}{
		{
			name: "V1 checkpoint",
			currentCheckpoint: &MemoryManagerCheckpointV1{
				PolicyName: "none",
				MachineState: NUMANodeMap{
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
				Entries: ContainerMemoryAssignments{},
			},
			expectedLegacyChecksum: func() checksum.Checksum {
				type MemoryManagerCheckpoint struct {
					PolicyName   string                     `json:"policyName"`
					MachineState NUMANodeMap                `json:"machineState"`
					Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
					Checksum     checksum.Checksum          `json:"checksum"`
				}

				return checksum.New(&MemoryManagerCheckpoint{
					PolicyName: "none",
					MachineState: NUMANodeMap{
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
					Entries: ContainerMemoryAssignments{},
				})
			},
		},
		{
			name: "V2 checkpoint",
			currentCheckpoint: &MemoryManagerCheckpointV2{
				PolicyName: "none",
				MachineState: NUMANodeMap{
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
				Entries:    ContainerMemoryAssignments{},
				PodEntries: PodMemoryAssignments{},
			},
			expectedLegacyChecksum: func() checksum.Checksum {
				type MemoryManagerCheckpoint struct {
					PolicyName   string                     `json:"policyName"`
					MachineState NUMANodeMap                `json:"machineState"`
					Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
					PodEntries   PodMemoryAssignments       `json:"podEntries,omitempty"`
					Checksum     checksum.Checksum          `json:"checksum"`
				}

				return checksum.New(&MemoryManagerCheckpoint{
					PolicyName: "none",
					MachineState: NUMANodeMap{
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
					Entries:    ContainerMemoryAssignments{},
					PodEntries: PodMemoryAssignments{},
				})
			},
		},
		{
			name: "V3 checkpoint",
			currentCheckpoint: &MemoryManagerCheckpointV3{
				MemoryManagerCheckpointV1: MemoryManagerCheckpointV1{
					PolicyName: "none",
					MachineState: NUMANodeMap{
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
					Entries: ContainerMemoryAssignments{},
				},
				CheckpointData: MemoryManagerCheckpointData{
					PolicyName: "none",
					MachineState: NUMANodeMap{
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
					Entries:    ContainerMemoryAssignments{},
					PodEntries: PodMemoryAssignments{},
				},
			},
			expectedLegacyChecksum: func() checksum.Checksum {
				type MemoryManagerCheckpoint struct {
					PolicyName   string                     `json:"policyName"`
					MachineState NUMANodeMap                `json:"machineState"`
					Entries      ContainerMemoryAssignments `json:"entries,omitempty"`
					Checksum     checksum.Checksum          `json:"checksum"`
				}

				return checksum.New(&MemoryManagerCheckpoint{
					PolicyName: "none",
					MachineState: NUMANodeMap{
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
					Entries: ContainerMemoryAssignments{},
				})
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Marshal the checkpoint using the logic that forces the "MemoryManagerCheckpoint" name
			data, err := tc.currentCheckpoint.(interface{ MarshalCheckpoint() ([]byte, error) }).MarshalCheckpoint()
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

			// 3. Compute the expected legacy checksum
			expectedLegacyChecksum := tc.expectedLegacyChecksum()

			// 4. Assert that the checksum written by our 1.37+ code matches
			// what an older Kubelet would expect to see.
			if writtenChecksum != expectedLegacyChecksum {
				t.Errorf("Written Checksum %d does not match legacy calculation %d. Forward compatibility broken.", writtenChecksum, expectedLegacyChecksum)
			}
		})
	}
}

func TestMemoryManagerCheckpoint_RoundTrip(t *testing.T) {
	testCases := []struct {
		name     string
		original checkpointmanager.Checkpoint
		restored checkpointmanager.Checkpoint
		verify   func(t *testing.T, original, restored checkpointmanager.Checkpoint)
	}{
		{
			name: "V1 checkpoint",
			original: &MemoryManagerCheckpointV1{
				PolicyName: "static",
				MachineState: NUMANodeMap{
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
				Entries: ContainerMemoryAssignments{
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
			},
			restored: newMemoryManagerCheckpointV1(),
			verify: func(t *testing.T, original, restored checkpointmanager.Checkpoint) {
				o := original.(*MemoryManagerCheckpointV1)
				r := restored.(*MemoryManagerCheckpointV1)
				require.Equal(t, o.PolicyName, r.PolicyName)
				require.Equal(t, o.MachineState, r.MachineState)
				require.Equal(t, o.Entries, r.Entries)
			},
		},
		{
			name: "V2 checkpoint",
			original: &MemoryManagerCheckpointV2{
				PolicyName: "static",
				MachineState: NUMANodeMap{
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
				Entries: ContainerMemoryAssignments{
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
				PodEntries: PodMemoryAssignments{
					"pod": PodEntry{
						MemoryBlocks: []Block{
							{
								NUMAAffinity: []int{0},
								Type:         v1.ResourceMemory,
								Size:         512,
							},
						},
					},
				},
			},
			restored: newMemoryManagerCheckpointV2(),
			verify: func(t *testing.T, original, restored checkpointmanager.Checkpoint) {
				o := original.(*MemoryManagerCheckpointV2)
				r := restored.(*MemoryManagerCheckpointV2)
				require.Equal(t, o.PolicyName, r.PolicyName)
				require.Equal(t, o.MachineState, r.MachineState)
				require.Equal(t, o.Entries, r.Entries)
				require.Equal(t, o.PodEntries, r.PodEntries)
			},
		},
		{
			name: "V3 checkpoint",
			original: &MemoryManagerCheckpointV3{
				MemoryManagerCheckpointV1: MemoryManagerCheckpointV1{
					PolicyName: "static",
					MachineState: NUMANodeMap{
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
					Entries: ContainerMemoryAssignments{
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
				},
				CheckpointData: MemoryManagerCheckpointData{
					PolicyName: "static",
					MachineState: NUMANodeMap{
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
					Entries: ContainerMemoryAssignments{
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
					PodEntries: PodMemoryAssignments{
						"pod": PodEntry{
							MemoryBlocks: []Block{
								{
									NUMAAffinity: []int{0},
									Type:         v1.ResourceMemory,
									Size:         512,
								},
							},
						},
					},
				},
			},
			restored: newMemoryManagerCheckpointV3(),
			verify: func(t *testing.T, original, restored checkpointmanager.Checkpoint) {
				o := original.(*MemoryManagerCheckpointV3)
				r := restored.(*MemoryManagerCheckpointV3)
				// embedded V2 part
				require.Equal(t, o.PolicyName, r.PolicyName)
				require.Equal(t, o.MachineState, r.MachineState)
				require.Equal(t, o.Entries, r.Entries)
				// V4 part
				require.Equal(t, o.CheckpointData.PolicyName, r.CheckpointData.PolicyName)
				require.Equal(t, o.CheckpointData.MachineState, r.CheckpointData.MachineState)
				require.Equal(t, o.CheckpointData.Entries, r.CheckpointData.Entries)
				require.Equal(t, o.CheckpointData.PodEntries, r.CheckpointData.PodEntries)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			data, err := tc.original.MarshalCheckpoint()
			require.NoError(t, err)

			require.NoError(t, tc.restored.UnmarshalCheckpoint(data))
			require.NoError(t, tc.restored.VerifyChecksum())

			tc.verify(t, tc.original, tc.restored)
		})
	}
}
