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
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	testutil "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state/testing"
	"k8s.io/utils/cpuset"
)

const (
	testingCheckpoint = "cpumanager_checkpoint_test"
	stateDir          = "cpumanager_state_test"
)

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
				"policyName": "static",
				"defaultCPUSet": "7-9",
				"entries": {
					"pod": {
						"container1": "4-6",
						"container2": "1-3"
					}
				},
				"checksum": 1942532442,
				"dataChecksum": 1234
			}`,
			"none",
			containermap.ContainerMap{},
			"could not load v4 checkpoint: failed to deserialize cpu manager checkpoint data: unexpected end of JSON input",
			nil,
		},
		{
			"Fail to restore checkpoint without dataChecksum section",
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
				"checksum": 1942532442,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"7-9\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"1-3\"}}}"
			}`,
			"none",
			containermap.ContainerMap{},
			"could not load v4 checkpoint: checkpoint is corrupted",
			nil,
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore default CPU set",
			nil,
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"4-6\"}",
				"dataChecksum": 657950972
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				defaultCPUSet: cpuset.New(4, 5, 6),
			},
		},
		// In below test case V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore valid checkpoint with InPlacePodVerticalScalingExclusiveCPUs disabled",
			FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: false},
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"7-9\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"1-3\"}}}",
				"dataChecksum": 1420829534
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
		// In below test case V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore valid checkpoint with InPlacePodVerticalScalingExclusiveCPUs enabled",
			FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: true},
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"7-9\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"1-3\"}}}",
				"dataChecksum": 1420829534
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
				baselines: ContainerCPUBaselines{
					"pod": {
						"container1": {
							Baseline: cpuset.New(4, 5, 6),
						},
						"container2": {
							Baseline: cpuset.New(1, 2, 3),
						},
					},
				},
			},
		},
		{
			"Fail to restore checkpoint with invalid checksum",
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
				"checksum": 1942532442,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"7-9\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"1-3\"}}}",
				"dataChecksum": 1234
			}`,
			"static",
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
		// In below test case policyName in V2 part of checkpoint is intentionally unaligned with data section.
		{
			"Fail to restore checkpoint with invalid policy name",
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
				"checksum": 1942532442,
				"data": "{\"policyName\":\"other\",\"defaultCPUSet\":\"7-9\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"1-3\"}}}",
				"dataChecksum": 1432271708
			}`,
			"none",
			containermap.ContainerMap{},
			`configured policy "none" differs from state checkpoint policy "other"`,
			nil,
		},
		// In below testcase defaultCPUSet in V2 part of checkpoint is intentionally unaligned with data section.
		{
			"Fail to restore checkpoint with unparsable default CPU set",
			nil,
			`{
				"policyName": "none",
				"defaultCPUSet": "7-9",
				"checksum": 98002435,
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"1.3\"}",
				"dataChecksum": 3033143655
			}`,
			"none",
			containermap.ContainerMap{},
			`could not parse default CPU set "1.3": strconv.Atoi: parsing "1.3": invalid syntax`,
			nil,
		},
		// In below testcase entries in V2 part of checkpoint are intentionally unaligned with data section.
		{
			"Fail to restore checkpoint with unparsable assignment entry",
			nil,
			`{
				"policyName": "static",
				"defaultCPUSet": "1-3",
				"entries": {
					"pod": {
						"container1": "4-6",
						"container2": "7-9"
					}
				},
				"checksum": 1659543133,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"asd\"}}}",
				"dataChecksum": 3794806925
			}`,
			"static",
			containermap.ContainerMap{},
			`could not parse cpuset "asd" for container "container2" in pod "pod": strconv.Atoi: parsing "asd": invalid syntax`,
			nil,
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore checkpoint ignoring unknown fields in data section",
			nil,
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"none\",\"defaultCPUSet\":\"4-6\",\"unknownField\":\"value\"}",
				"dataChecksum": 3492408555
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				defaultCPUSet: cpuset.New(4, 5, 6),
			},
		},
		{
			"Fail to restore checkpoint from checkpoint with v1 checksum",
			nil,
			`{
				"policyName": "none",
				"defaultCPUSet": "1-3",
				"checksum": 1694838852
			}`,
			"none",
			containermap.ContainerMap{},
			"checkpoint is corrupted",
			nil,
		},
		{
			"Fail to restore checkpoint from v1 (migration)",
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
			containermap.ContainerMap{},
			"cannot unmarshal string into Go struct field CPUManagerCheckpointV2.entries",
			nil,
		},
		{
			"Restore checkpoint from v2 (migration) with InPlacePodVerticalScalingExclusiveCPUs disabled",
			FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: false},
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
			"Restore checkpoint from v2 (migration) with InPlacePodVerticalScalingExclusiveCPUs enabled",
			FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: true},
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
				baselines: ContainerCPUBaselines{
					"pod": {
						"container1": {
							Baseline: cpuset.New(4, 5, 6),
						},
						"container2": {
							Baseline: cpuset.New(1, 2, 3),
						},
					},
				},
			},
		},
		{
			"Restore checkpoint from v3 (migration) with PodLevelResourceManagers disabled and InPlacePodVerticalScalingExclusiveCPUs disabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               false,
				features.InPlacePodVerticalScalingExclusiveCPUs: false,
			},
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
				"checksum": 766259872
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
			"Restore checkpoint from v3 (migration) with PodLevelResourceManagers disabled and InPlacePodVerticalScalingExclusiveCPUs enabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               false,
				features.InPlacePodVerticalScalingExclusiveCPUs: true,
			},
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
				"checksum": 766259872
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
				baselines: ContainerCPUBaselines{
					"pod1": {
						"container1": {
							Baseline: cpuset.New(5, 6),
						},
						"container2": {
							Baseline: cpuset.New(3, 4),
						},
					},
				},
			},
		},
		{
			"Restore checkpoint from v3 (migration) with PodLevelResourceManagers enabled and InPlacePodVerticalScalingExclusiveCPUs disabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               true,
				features.InPlacePodVerticalScalingExclusiveCPUs: false,
			},
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
				"checksum": 766259872
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
			"Restore checkpoint from v3 (migration) with PodLevelResourceManagers enabled and InPlacePodVerticalScalingExclusiveCPUs enabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               true,
				features.InPlacePodVerticalScalingExclusiveCPUs: true,
			},
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
				"checksum": 766259872
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
				baselines: ContainerCPUBaselines{
					"pod1": {
						"container1": {
							Baseline: cpuset.New(5, 6),
						},
						"container2": {
							Baseline: cpuset.New(3, 4),
						},
					},
				},
			},
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore valid v4 checkpoint with PodLevelResourceManagers disabled and InPlacePodVerticalScalingExclusiveCPUs disabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               false,
				features.InPlacePodVerticalScalingExclusiveCPUs: false,
			},
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-10\"}},\"baselines\":{\"pod\":{\"container1\":{\"baseline\":\"4-5\"},\"container2\":{\"baseline\":\"7-8\"}}}}",
				"dataChecksum": 1032795074
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
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore valid v4 checkpoint with PodLevelResourceManagers disabled and InPlacePodVerticalScalingExclusiveCPUs enabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               false,
				features.InPlacePodVerticalScalingExclusiveCPUs: true,
			},
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-10\"}},\"baselines\":{\"pod\":{\"container1\":{\"baseline\":\"4-5\"},\"container2\":{\"baseline\":\"7-8\"}}}}",
				"dataChecksum": 1032795074
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
				baselines: ContainerCPUBaselines{
					"pod": {
						"container1": {
							Baseline: cpuset.New(4, 5),
						},
						"container2": {
							Baseline: cpuset.New(7, 8),
						},
					},
				},
			},
		},
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore valid v4 checkpoint with PodLevelResourceManagers enabled and InPlacePodVerticalScalingExclusiveCPUs disabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               true,
				features.InPlacePodVerticalScalingExclusiveCPUs: false,
			},
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-10\"}},\"baselines\":{\"pod\":{\"container1\":{\"baseline\":\"4-5\"},\"container2\":{\"baseline\":\"7-8\"}}}}",
				"dataChecksum": 1032795074
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
		// In below testcase V2 part of checkpoint is intentionally corrupted to verify that it is not used.
		{
			"Restore valid v4 checkpoint with PodLevelResourceManagers enabled and InPlacePodVerticalScalingExclusiveCPUs enabled",
			FeatureGateCombination{
				features.PodLevelResourceManagers:               true,
				features.InPlacePodVerticalScalingExclusiveCPUs: true,
			},
			`{
				"policyName": "other",
				"defaultCPUSet": "1-9",
				"checksum": 1234,
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-10\"}},\"baselines\":{\"pod\":{\"container1\":{\"baseline\":\"4-5\"},\"container2\":{\"baseline\":\"7-8\"}}}}",
				"dataChecksum": 1032795074
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
				baselines: ContainerCPUBaselines{
					"pod": {
						"container1": {
							Baseline: cpuset.New(4, 5),
						},
						"container2": {
							Baseline: cpuset.New(7, 8),
						},
					},
				},
			},
		},
		// The pending resize changes both a container and the pod-level assignment.
		// PodLevelResourceManagers does not support in-place resize yet, so the scenario is artificial.
		{
			"Restore a pending resize with InPlacePodVerticalScalingExclusiveCPUs enabled",
			FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: true},
			`{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"baselines\":{\"pod\":{\"container1\":{\"baseline\":\"4-5\"},\"container2\":{\"baseline\":\"7-8\"}}},\"pendingResize\":{\"podUID\":\"pod\",\"containerCpuSets\":{\"container1\":\"4-5\"},\"podCpuSet\":\"4-9\"}}",
				"dataChecksum": 2220107972
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
				baselines: ContainerCPUBaselines{
					"pod": {
						"container1": {
							Baseline: cpuset.New(4, 5),
						},
						"container2": {
							Baseline: cpuset.New(7, 8),
						},
					},
				},
				pendingResize: &PendingResize{
					PodUID: "pod",
					ContainerCPUSets: map[string]cpuset.CPUSet{
						"container1": cpuset.New(4, 5),
					},
					PodCPUSet: new(cpuset.New(4, 5, 6, 7, 8, 9)),
				},
			},
		},
		{
			"Ignore a pending resize with InPlacePodVerticalScalingExclusiveCPUs disabled",
			FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: false},
			`{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container1\":\"4-6\",\"container2\":\"7-9\"}},\"baselines\":{\"pod\":{\"container1\":{\"baseline\":\"4-5\"},\"container2\":{\"baseline\":\"7-8\"}}},\"pendingResize\":{\"podUID\":\"pod\",\"containerCpuSets\":{\"container1\":\"4-5\"},\"podCpuSet\":\"4-9\"}}",
				"dataChecksum": 2220107972
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
		features.InPlacePodVerticalScalingExclusiveCPUs,
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
		description    string
		fgRequirements FeatureGateCombination
		expectedState  *stateMemory
	}{
		{
			description:    "Store default cpu set",
			fgRequirements: nil,
			expectedState:  &stateMemory{defaultCPUSet: cpuset.New(1, 2, 3)},
		},
		{
			description:    "Store assignments",
			fgRequirements: FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: false},
			expectedState: &stateMemory{
				assignments: map[string]map[string]cpuset.CPUSet{
					"pod": {
						"container1": cpuset.New(1, 5, 8),
					},
				},
			},
		},
		{
			description:    "Store baselines",
			fgRequirements: FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: true},
			expectedState: &stateMemory{
				assignments: map[string]map[string]cpuset.CPUSet{
					"pod": {
						"container1": cpuset.New(1, 5, 8),
					},
				},
				baselines: ContainerCPUBaselines{
					"pod": {
						"container1": {
							Baseline: cpuset.New(1, 5),
						},
					},
				},
			},
		},
		{
			description:    "Store a pending resize",
			fgRequirements: FeatureGateCombination{features.InPlacePodVerticalScalingExclusiveCPUs: true},
			expectedState: &stateMemory{
				assignments: map[string]map[string]cpuset.CPUSet{
					"pod": {
						"container1": cpuset.New(1, 5, 8),
					},
				},
				baselines: ContainerCPUBaselines{
					"pod": {
						"container1": {
							Baseline: cpuset.New(1, 5),
						},
					},
				},
				pendingResize: &PendingResize{
					PodUID: "pod",
					ContainerCPUSets: map[string]cpuset.CPUSet{
						"container1": cpuset.New(1, 5, 8, 9),
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

	// list of all features verified in this test
	featureGateList := []featuregate.Feature{
		features.PodLevelResourceManagers,
		features.InPlacePodVerticalScalingExclusiveCPUs,
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

					logger, _ := ktesting.NewTestContext(t)
					cs1, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
					if err != nil {
						t.Fatalf("could not create testing checkpointState instance: %v", err)
					}

					// set values of cs1 instance so they are stored in checkpoint and can be read by cs2
					cs1.SetDefaultCPUSet(tc.expectedState.defaultCPUSet)
					cs1.SetCPUAssignments(tc.expectedState.assignments)
					cs1.SetCPUBaselines(tc.expectedState.baselines)
					if tc.expectedState.pendingResize != nil {
						cs1.SetPendingResize(*tc.expectedState.pendingResize)
					}

					// restore checkpoint with previously stored values
					cs2, err := NewCheckpointState(logger, testingDir, testingCheckpoint, "none", nil)
					if err != nil {
						t.Fatalf("could not create testing checkpointState instance: %v", err)
					}

					AssertStateEqual(t, cs2, tc.expectedState)
				})
			}
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
			err = cpm.RemoveCheckpoint(testingCheckpoint)
			require.NoErrorf(t, err, "could not remove previous checkpoint: %v", err)

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

	cpubaselinesSf := sf.GetCPUBaselines()
	cpubaselinesSm := sm.GetCPUBaselines()
	if !reflect.DeepEqual(cpubaselinesSf, cpubaselinesSm) {
		t.Errorf("State CPU baseline entries mismatch. Have %s, want %s", cpubaselinesSf, cpubaselinesSm)
	}

	pendingSf, foundSf := sf.GetPendingResize()
	pendingSm, foundSm := sm.GetPendingResize()
	if foundSf != foundSm || !reflect.DeepEqual(pendingSf, pendingSm) {
		t.Errorf("State pending resize mismatch. Have %v (found %t), want %v (found %t)", pendingSf, foundSf, pendingSm, foundSm)
	}

}

func TestCPUManagerCheckpoint_MarshalCheckpoint_HashCompatibility(t *testing.T) {
	testCases := []struct {
		name                   string
		currentCheckpoint      any
		expectedLegacyChecksum func() checksum.Checksum
	}{
		{
			name: "V2 checkpoint",
			currentCheckpoint: &CPUManagerCheckpointV2{
				PolicyName:    "none",
				DefaultCPUSet: "1-3",
				Entries:       make(map[string]map[string]string),
			},
			expectedLegacyChecksum: func() checksum.Checksum {
				type CPUManagerCheckpoint struct {
					PolicyName    string                       `json:"policyName"`
					DefaultCPUSet string                       `json:"defaultCpuSet"`
					Entries       map[string]map[string]string `json:"entries,omitempty"`
					Checksum      checksum.Checksum            `json:"checksum"`
				}
				return checksum.New(&CPUManagerCheckpoint{
					PolicyName:    "none",
					DefaultCPUSet: "1-3",
					Entries:       make(map[string]map[string]string),
				})
			},
		},
		{
			name: "V3 checkpoint",
			currentCheckpoint: &CPUManagerCheckpointV3{
				PolicyName:    "none",
				DefaultCPUSet: "1-3",
				Entries:       make(map[string]map[string]string),
				PodEntries:    PodCPUAssignments{"pod": PodEntry{cpuset.New(4, 5)}},
			},
			expectedLegacyChecksum: func() checksum.Checksum {
				type CPUManagerCheckpoint struct {
					PolicyName    string                       `json:"policyName"`
					DefaultCPUSet string                       `json:"defaultCpuSet"`
					Entries       map[string]map[string]string `json:"entries,omitempty"`
					PodEntries    PodCPUAssignments            `json:"podEntries,omitempty"`
					Checksum      checksum.Checksum            `json:"checksum"`
				}
				return checksum.New(&CPUManagerCheckpoint{
					PolicyName:    "none",
					DefaultCPUSet: "1-3",
					Entries:       make(map[string]map[string]string),
					PodEntries:    PodCPUAssignments{"pod": PodEntry{cpuset.New(4, 5)}},
				})
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Marshal the checkpoint using the logic that forces the "CPUManagerCheckpoint" name
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

			// 4. Assert that the checksum written by our 1.36+ code matches
			// what a 1.35 Kubelet would expect to see.
			if writtenChecksum != expectedLegacyChecksum {
				t.Errorf("Written Checksum %d does not match legacy calculation %d. Forward compatibility broken.", writtenChecksum, expectedLegacyChecksum)
			}
		})
	}
}

func TestCPUManagerCheckpointV3_VerifyChecksum_Compatibility(t *testing.T) {
	testCases := []struct {
		name     string
		checksum checksum.Checksum
	}{
		{
			name:     "accepts legacy v3 checksum",
			checksum: 766259872,
		},
		{
			name:     "accepts current v3 checksum",
			checksum: 2284712151,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cp := &CPUManagerCheckpointV3{
				PolicyName:    "static",
				DefaultCPUSet: "1-2",
				Entries: map[string]map[string]string{
					"pod1": {
						"container1": "5-6",
						"container2": "3-4",
					},
				},
				PodEntries: PodCPUAssignments{
					"pod2": {CPUSet: cpuset.New(7, 8, 9)},
				},
				Checksum: tc.checksum,
			}

			require.NoError(t, cp.VerifyChecksum())
		})
	}
}

func TestCPUManagerCheckpoint_RoundTrip(t *testing.T) {
	testCases := []struct {
		name     string
		original checkpointmanager.Checkpoint
		restored checkpointmanager.Checkpoint
		verify   func(t *testing.T, original, restored checkpointmanager.Checkpoint)
	}{
		{
			name: "V2 checkpoint",
			original: &CPUManagerCheckpointV2{
				PolicyName:    "static",
				DefaultCPUSet: "0-3",
				Entries: map[string]map[string]string{
					"pod1": {"container1": "4-5"},
				},
			},
			restored: newCPUManagerCheckpointV2(),
			verify: func(t *testing.T, original, restored checkpointmanager.Checkpoint) {
				o := original.(*CPUManagerCheckpointV2)
				r := restored.(*CPUManagerCheckpointV2)
				require.Equal(t, o.PolicyName, r.PolicyName)
				require.Equal(t, o.DefaultCPUSet, r.DefaultCPUSet)
				require.Equal(t, o.Entries, r.Entries)
			},
		},
		{
			name: "V3 checkpoint",
			original: &CPUManagerCheckpointV3{
				PolicyName:    "static",
				DefaultCPUSet: "0-3",
				Entries: map[string]map[string]string{
					"pod1": {"container1": "4-5"},
				},
				PodEntries: PodCPUAssignments{
					"pod2": {CPUSet: cpuset.New(6, 7)},
				},
			},
			restored: newCPUManagerCheckpointV3(),
			verify: func(t *testing.T, original, restored checkpointmanager.Checkpoint) {
				o := original.(*CPUManagerCheckpointV3)
				r := restored.(*CPUManagerCheckpointV3)
				require.Equal(t, o.PolicyName, r.PolicyName)
				require.Equal(t, o.DefaultCPUSet, r.DefaultCPUSet)
				require.Equal(t, o.Entries, r.Entries)
				require.Equal(t, o.PodEntries, r.PodEntries)
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

func removeAll(dir string, t *testing.T) {
	t.Helper()
	if err := os.RemoveAll(dir); err != nil {
		t.Fatalf("unable to remove dir %s: %v", dir, err)
	}
}

func TestSetPodCPUSet(t *testing.T) {
	testCases := []struct {
		description        string
		featureGateEnabled bool
		cpuSet             []int
		expected           string
	}{
		{
			description:        "Assign CPUSet to given pod with PodLevelResourceManagers feature gate enabled",
			featureGateEnabled: true,
			cpuSet:             []int{1, 2, 3, 4, 8},
			expected:           "1-4,8",
		},
		{
			description:        "Assign CPUSet to given pod with PodLevelResourceManagers feature gate disabled",
			featureGateEnabled: false,
			cpuSet:             []int{1, 2, 3, 4, 8},
			expected:           "",
		},
	}

	logger, _ := ktesting.NewTestContext(t)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, testCase.featureGateEnabled)

			tmpDir, err := os.MkdirTemp("", stateDir)
			require.NoError(t, err)

			// Remove previous checkpoint's dir during subtest cleanup
			t.Cleanup(func() {
				require.NoErrorf(t, os.RemoveAll(tmpDir), "unable to remove dir %s", tmpDir)
			})

			state1, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			state1.SetPodCPUSet("pod1", cpuset.New(testCase.cpuSet...))

			state2, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			actual, _ := state2.GetPodCPUSet("pod1")
			require.Equal(t, testCase.expected, actual.String())
		})
	}
}

func TestSetPodCPUAssignments(t *testing.T) {
	testCases := []struct {
		description        string
		featureGateEnabled bool
		podEntries         PodCPUAssignments
		expected           map[string]string
	}{
		{
			description:        "Assign CPUSet to pods with PodLevelResourceManagers feature gate enabled",
			featureGateEnabled: true,
			podEntries: PodCPUAssignments{
				"pod1": {CPUSet: cpuset.New(4, 7)},
				"pod2": {CPUSet: cpuset.New(1, 2, 3)},
			},
			expected: map[string]string{
				"pod1": "4,7",
				"pod2": "1-3",
			},
		},
		{
			description:        "Assign CPUSet to pods with PodLevelResourceManagers feature gate disabled",
			featureGateEnabled: false,
			podEntries: PodCPUAssignments{
				"pod1": {CPUSet: cpuset.New(3, 5)},
			},
			expected: map[string]string{},
		},
	}

	logger, _ := ktesting.NewTestContext(t)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, testCase.featureGateEnabled)

			tmpDir, err := os.MkdirTemp("", stateDir)
			require.NoError(t, err)

			// Remove previous checkpoint's dir during subtest cleanup
			t.Cleanup(func() {
				require.NoErrorf(t, os.RemoveAll(tmpDir), "unable to remove dir %s", tmpDir)
			})

			state1, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			state1.SetPodCPUAssignments(testCase.podEntries)

			state2, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			actual := state2.GetPodCPUAssignments()
			require.Len(t, actual, len(testCase.expected))

			for pod, entry := range actual {
				require.Equal(t, testCase.expected[pod], entry.CPUSet.String())
			}
		})
	}
}

func TestGetBaselineCPUSet(t *testing.T) {
	testCases := []struct {
		description        string
		featureGateEnabled bool
		content            string
		expected           ContainerCPUBaselines
	}{
		{
			description:        "Get baseline CPUSet with InPlacePodVerticalScalingExclusiveCPUs feature gate enabled when baselines field included",
			featureGateEnabled: true,
			content: `{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container\":\"4-5\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-6\"}},\"baselines\":{\"pod\":{\"container\":{\"baseline\":\"5\"}}}}",
				"dataChecksum": 4038000775
			}`,
			expected: ContainerCPUBaselines{
				"pod": {
					"container": {
						Baseline: cpuset.New(5),
					},
				},
			},
		},
		{
			description:        "Get baseline CPUSet with InPlacePodVerticalScalingExclusiveCPUs feature gate enabled when baselines field excluded",
			featureGateEnabled: true,
			content: `{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container\":\"4-5\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-6\"}}}",
				"dataChecksum": 234652157
			}`,
			expected: ContainerCPUBaselines{
				"pod": {
					"container": {
						Baseline: cpuset.New(4, 5), // Must be equal to value from entries
					},
				},
			},
		},
		{
			description:        "Get baseline CPUSet with InPlacePodVerticalScalingExclusiveCPUs feature gate disabled",
			featureGateEnabled: false,
			content: `{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod\":{\"container\":\"4-5\"}},\"podEntries\":{\"pod\":{\"cpuSet\":\"4-6\"}}}",
				"dataChecksum": 234652157
			}`,
			expected: ContainerCPUBaselines{
				"pod": {
					"container": {
						Baseline: cpuset.New(),
					},
				},
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	tmpDir, err := os.MkdirTemp("", stateDir)
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(tmpDir), "unable to remove dir %s", tmpDir)
	})

	cpm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	require.NoErrorf(t, err, "could not create testing checkpoint manager: %v", err)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, testCase.featureGateEnabled)

			err = cpm.RemoveCheckpoint(testingCheckpoint)
			require.NoErrorf(t, err, "could not remove previous checkpoint: %v", err)

			checkpoint := &testutil.MockCheckpoint{Content: testCase.content}
			err = cpm.CreateCheckpoint(testingCheckpoint, checkpoint)
			require.NoErrorf(t, err, "could not create testing checkpoint: %v", err)

			state, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			originals, _ := state.GetBaselineCPUSet("pod", "container")
			require.Equal(t, testCase.expected["pod"]["container"].Baseline, originals)
		})
	}
}

func TestDeletePod(t *testing.T) {
	testCases := []struct {
		description        string
		featureGateEnabled bool
		content            string
		podUIDs            []string // Pods to delete
		expected           map[string]string
	}{
		{
			description:        "Delete existing pod(s) CPUSet with PodLevelResourceManagers feature gate enabled",
			featureGateEnabled: true,
			content: `{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod1\":{\"container\":\"4-5\"},\"pod2\":{\"container\":\"7-8\"}},\"podEntries\":{\"pod1\":{\"cpuSet\":\"4-6\"},\"pod2\":{\"cpuSet\":\"7-9\"}}}",
				"dataChecksum": 2086345991
			}`,
			podUIDs: []string{"pod1"},
			expected: map[string]string{
				"pod2": "7-9",
			},
		},
		{
			description:        "Delete nonexisting pod(s) CPUSet with PodLevelResourceManagers feature gate enabled",
			featureGateEnabled: true,
			content: `{
				"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3\",\"entries\":{\"pod1\":{\"container\":\"4-5\"},\"pod2\":{\"container\":\"7-8\"}},\"podEntries\":{\"pod1\":{\"cpuSet\":\"4-6\"},\"pod2\":{\"cpuSet\":\"7-9\"}}}",
				"dataChecksum": 2086345991
			}`,
			podUIDs: []string{"pod3", "pod4"},
			expected: map[string]string{
				"pod1": "4-6",
				"pod2": "7-9",
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	tmpDir, err := os.MkdirTemp("", stateDir)
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(tmpDir), "unable to remove dir %s", tmpDir)
	})

	cpm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	require.NoErrorf(t, err, "could not create testing checkpoint manager: %v", err)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourceManagers, testCase.featureGateEnabled)

			err = cpm.RemoveCheckpoint(testingCheckpoint)
			require.NoErrorf(t, err, "could not remove previous checkpoint: %v", err)

			checkpoint := &testutil.MockCheckpoint{Content: testCase.content}
			err = cpm.CreateCheckpoint(testingCheckpoint, checkpoint)
			require.NoErrorf(t, err, "could not create testing checkpoint: %v", err)

			state, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			for _, uid := range testCase.podUIDs {
				state.DeletePod(uid)
			}
			actual := state.GetPodCPUAssignments()

			require.Len(t, actual, len(testCase.expected))
			for pod, entry := range actual {
				require.Equal(t, testCase.expected[pod], entry.CPUSet.String())
			}
		})
	}
}

// pendingResizeCheckpoint has two pods with one container each, and a pending
// resize of pod1 that grows its container from {4,5} to {4,5,6}.
const pendingResizeCheckpoint = `{
	"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3,6\",\"entries\":{\"pod1\":{\"container\":\"4-5\"},\"pod2\":{\"container\":\"7-8\"}},\"pendingResize\":{\"podUID\":\"pod1\",\"containerCpuSets\":{\"container\":\"4-6\"}}}",
	"dataChecksum": 212795483
}`

// noPendingResizeCheckpoint is pendingResizeCheckpoint without the pending resize.
const noPendingResizeCheckpoint = `{
	"data": "{\"policyName\":\"static\",\"defaultCPUSet\":\"1-3,6\",\"entries\":{\"pod1\":{\"container\":\"4-5\"},\"pod2\":{\"container\":\"7-8\"}}}",
	"dataChecksum": 3305408017
}`

// newPendingResizeTestCheckpoint returns a checkpoint manager for a temporary
// directory, removed when the test ends.
func newPendingResizeTestCheckpoint(t *testing.T) (string, checkpointmanager.CheckpointManager) {
	t.Helper()
	tmpDir, err := os.MkdirTemp("", stateDir)
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoErrorf(t, os.RemoveAll(tmpDir), "unable to remove dir %s", tmpDir)
	})
	cpm, err := checkpointmanager.NewCheckpointManager(tmpDir)
	require.NoErrorf(t, err, "could not create testing checkpoint manager: %v", err)
	return tmpDir, cpm
}

// writeTestCheckpoint replaces the testing checkpoint with content.
func writeTestCheckpoint(t *testing.T, cpm checkpointmanager.CheckpointManager, content string) {
	t.Helper()
	require.NoError(t, cpm.RemoveCheckpoint(testingCheckpoint), "could not remove previous checkpoint")
	require.NoError(t, cpm.CreateCheckpoint(testingCheckpoint, &testutil.MockCheckpoint{Content: content}), "could not create testing checkpoint")
}

// Test describes artificial scenario as pod-level exclusive CPUs are not scalable.
func TestSetPendingResize(t *testing.T) {
	testCases := []struct {
		description        string
		featureGateEnabled bool
		expectedFound      bool
	}{
		{
			description:        "Store the pending resize with InPlacePodVerticalScalingExclusiveCPUs feature gate enabled",
			featureGateEnabled: true,
			expectedFound:      true,
		},
		{
			description:        "Store the pending resize with InPlacePodVerticalScalingExclusiveCPUs feature gate disabled",
			featureGateEnabled: false,
			expectedFound:      false,
		},
	}

	logger, _ := ktesting.NewTestContext(t)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, testCase.featureGateEnabled)
			tmpDir, _ := newPendingResizeTestCheckpoint(t)

			state1, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			// The second pending resize replaces the first one.
			state1.SetPendingResize(PendingResize{
				PodUID:           "pod1",
				ContainerCPUSets: map[string]cpuset.CPUSet{"container": cpuset.New(1, 2)},
			})
			expected := PendingResize{
				PodUID:           "pod2",
				ContainerCPUSets: map[string]cpuset.CPUSet{"container": cpuset.New(4, 5, 6)},
				PodCPUSet:        new(cpuset.New(4, 5, 6, 7, 8, 9)),
			}
			state1.SetPendingResize(expected)

			state2, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			actual, found := state2.GetPendingResize()
			require.Equal(t, testCase.expectedFound, found)
			if found {
				require.Equal(t, expected, actual)
			}
		})
	}
}

// An empty pod-level target and an unchanged pod-level assignment must stay
// distinct through the checkpoint.
// Test describes artificial scenario as pod-level exclusive CPUs are not scalable.
func TestPendingResizePodCPUSetThroughCheckpoint(t *testing.T) {
	testCases := []struct {
		description string
		podCPUSet   *cpuset.CPUSet
	}{
		{
			description: "Unchanged pod-level assignment",
		},
		{
			description: "Empty pod-level target",
			podCPUSet:   new(cpuset.New()),
		},
		{
			description: "Non-empty pod-level target",
			podCPUSet:   new(cpuset.New(4, 5, 6)),
		},
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	logger, _ := ktesting.NewTestContext(t)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			tmpDir, _ := newPendingResizeTestCheckpoint(t)

			state1, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)
			expected := PendingResize{
				PodUID:           "pod",
				ContainerCPUSets: map[string]cpuset.CPUSet{"container": cpuset.New(4, 5)},
				PodCPUSet:        testCase.podCPUSet,
			}
			state1.SetPendingResize(expected)

			state2, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)
			actual, found := state2.GetPendingResize()
			require.True(t, found)
			require.Equal(t, expected, actual)
		})
	}
}

func TestGetPendingResize(t *testing.T) {
	testCases := []struct {
		description        string
		featureGateEnabled bool
		content            string
		expectedFound      bool
	}{
		{
			description:        "Get the pending resize with InPlacePodVerticalScalingExclusiveCPUs feature gate enabled when the pendingResize field is included",
			featureGateEnabled: true,
			content:            pendingResizeCheckpoint,
			expectedFound:      true,
		},
		{
			description:        "Get the pending resize with InPlacePodVerticalScalingExclusiveCPUs feature gate enabled when the pendingResize field is excluded",
			featureGateEnabled: true,
			content:            noPendingResizeCheckpoint,
			expectedFound:      false,
		},
		{
			description:        "Get the pending resize with InPlacePodVerticalScalingExclusiveCPUs feature gate disabled",
			featureGateEnabled: false,
			content:            pendingResizeCheckpoint,
			expectedFound:      false,
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	tmpDir, cpm := newPendingResizeTestCheckpoint(t)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, testCase.featureGateEnabled)
			writeTestCheckpoint(t, cpm, testCase.content)

			state, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			actual, found := state.GetPendingResize()
			require.Equal(t, testCase.expectedFound, found)
			if found {
				require.Equal(t, PendingResize{PodUID: "pod1", ContainerCPUSets: map[string]cpuset.CPUSet{"container": cpuset.New(4, 5, 6)}}, actual)
			}
		})
	}
}

func TestCommitPendingResize(t *testing.T) {
	testCases := []struct {
		description           string
		content               string
		expectedAssignments   map[string]string
		expectedDefaultCPUSet string
	}{
		{
			description: "Commit the pending resize",
			content:     pendingResizeCheckpoint,
			expectedAssignments: map[string]string{
				"pod1": "4-6",
				"pod2": "7-8",
			},
			expectedDefaultCPUSet: "1-3",
		},
		{
			description: "Commit without a pending resize",
			content:     noPendingResizeCheckpoint,
			expectedAssignments: map[string]string{
				"pod1": "4-5",
				"pod2": "7-8",
			},
			expectedDefaultCPUSet: "1-3,6",
		},
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	logger, _ := ktesting.NewTestContext(t)
	tmpDir, cpm := newPendingResizeTestCheckpoint(t)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			writeTestCheckpoint(t, cpm, testCase.content)

			state1, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			state1.CommitPendingResize()

			// The commit is stored in the checkpoint, so restore it to check what a kubelet restart sees.
			state2, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			for _, state := range []State{state1, state2} {
				for pod, expected := range testCase.expectedAssignments {
					actual, _ := state.GetCPUSet(pod, "container")
					require.Equal(t, expected, actual.String())
				}
				require.Equal(t, testCase.expectedDefaultCPUSet, state.GetDefaultCPUSet().String())
				_, found := state.GetPendingResize()
				require.False(t, found, "the commit must delete the pending resize")
			}
		})
	}
}

func TestDeletePendingResize(t *testing.T) {
	testCases := []struct {
		description string
		content     string
	}{
		{
			description: "Delete the pending resize",
			content:     pendingResizeCheckpoint,
		},
		{
			description: "Delete without a pending resize",
			content:     noPendingResizeCheckpoint,
		},
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	logger, _ := ktesting.NewTestContext(t)
	tmpDir, cpm := newPendingResizeTestCheckpoint(t)

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			writeTestCheckpoint(t, cpm, testCase.content)

			state1, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			state1.DeletePendingResize()

			// The deletion is stored in the checkpoint, so restore it to check what a kubelet restart sees.
			state2, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
			require.NoError(t, err)

			for _, state := range []State{state1, state2} {
				_, found := state.GetPendingResize()
				require.False(t, found, "the pending resize must be deleted")
				actual, _ := state.GetCPUSet("pod1", "container")
				require.Equal(t, "4-5", actual.String(), "deleting a pending resize must not apply it")
				require.Equal(t, "1-3,6", state.GetDefaultCPUSet().String(), "deleting a pending resize must not change the default CPU set")
			}
		})
	}
}

// checkpointV4WithoutPendingResize is the V4 checkpoint before adding `PendingResize`.
type checkpointV4WithoutPendingResize struct {
	Data           string            `json:"data"`
	DataChecksum   checksum.Checksum `json:"dataChecksum"`
	CheckpointData struct {
		PolicyName    string                              `json:"policyName"`
		DefaultCPUSet string                              `json:"defaultCpuSet"`
		Entries       map[string]map[string]string        `json:"entries,omitempty"`
		PodEntries    PodCPUAssignments                   `json:"podEntries,omitempty"`
		Baselines     map[string]map[string]ContainerCPUs `json:"baselines,omitempty"`
	} `json:"-"`
}

func (cp *checkpointV4WithoutPendingResize) MarshalCheckpoint() ([]byte, error) {
	return nil, fmt.Errorf("not used by the test")
}

func (cp *checkpointV4WithoutPendingResize) UnmarshalCheckpoint(blob []byte) error {
	if err := json.Unmarshal(blob, cp); err != nil {
		return err
	}
	return json.Unmarshal([]byte(cp.Data), &cp.CheckpointData)
}

func (cp *checkpointV4WithoutPendingResize) VerifyChecksum() error {
	return cp.DataChecksum.Verify(cp.Data)
}

// writePendingResizeCheckpoint stores, through the state and with
// InPlacePodVerticalScalingExclusiveCPUs enabled, a checkpoint where the
// container of pod has {4,5}, the default CPU set is {1,2,3,6}, and a pending
// resize grows the container to {4,5,6}.
func writePendingResizeCheckpoint(t *testing.T) (string, checkpointmanager.CheckpointManager) {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	logger, _ := ktesting.NewTestContext(t)
	tmpDir, cpm := newPendingResizeTestCheckpoint(t)
	s, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
	require.NoError(t, err)
	s.SetCPUSet("pod", "container", cpuset.New(4, 5))
	s.SetDefaultCPUSet(cpuset.New(1, 2, 3, 6))
	s.SetPendingResize(PendingResize{PodUID: "pod", ContainerCPUSets: map[string]cpuset.CPUSet{"container": cpuset.New(4, 5, 6)}})
	return tmpDir, cpm
}

func TestPendingResizeCheckpointWrittenWithoutFeatureGate(t *testing.T) {
	tmpDir, _ := writePendingResizeCheckpoint(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, false)
	logger, _ := ktesting.NewTestContext(t)

	s, err := NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
	require.NoError(t, err)
	// Force checkpoint to be overwritten while FG is off.
	s.SetDefaultCPUSet(s.GetDefaultCPUSet())

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, true)
	s, err = NewCheckpointState(logger, tmpDir, testingCheckpoint, "static", containermap.ContainerMap{})
	require.NoError(t, err)

	_, found := s.GetPendingResize()
	require.False(t, found, "a write with the feature gate disabled must drop the pending resize")
	cset, _ := s.GetCPUSet("pod", "container")
	require.Equal(t, "4-5", cset.String())
	require.Equal(t, "1-3,6", s.GetDefaultCPUSet().String())
}

// After a rollback, a kubelet with V4 that predates pending resizes reads the
// checkpoint without the pendingResize field. The feature gate only decides
// whether such a kubelet restores the baselines after decoding, so it does not
// matter for reading the checkpoint.
func TestPendingResizeCheckpointReadByV4WithoutPendingResizes(t *testing.T) {
	_, cpm := writePendingResizeCheckpoint(t)

	cp := &checkpointV4WithoutPendingResize{}
	require.NoError(t, cpm.GetCheckpoint(testingCheckpoint, cp))

	require.Equal(t, "static", cp.CheckpointData.PolicyName)
	require.Equal(t, "1-3,6", cp.CheckpointData.DefaultCPUSet)
	require.Equal(t, map[string]map[string]string{"pod": {"container": "4-5"}}, cp.CheckpointData.Entries)
	require.Equal(t, map[string]map[string]ContainerCPUs{"pod": {"container": {Baseline: "4-5"}}}, cp.CheckpointData.Baselines)
}

// After a rollback, a kubelet that predates V4 reads only the embedded V2
// fields, which have their own checksum.
func TestPendingResizeCheckpointReadByV2(t *testing.T) {
	_, cpm := writePendingResizeCheckpoint(t)

	cp := newCPUManagerCheckpointV2()
	require.NoError(t, cpm.GetCheckpoint(testingCheckpoint, cp))

	// An empty checksum is always accepted, so make sure it was verified.
	require.NotZero(t, cp.Checksum, "the checkpoint must carry a V2 checksum")
	require.Equal(t, "static", cp.PolicyName)
	require.Equal(t, "1-3,6", cp.DefaultCPUSet)
	require.Equal(t, map[string]map[string]string{"pod": {"container": "4-5"}}, cp.Entries)
}
