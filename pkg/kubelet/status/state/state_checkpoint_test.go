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

package state

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
)

func newTestStateCheckpoint(t *testing.T) *stateCheckpoint {
	// create temp dir
	testingDir, err := os.MkdirTemp("", "pod_resource_allocation_state_test")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll(testingDir); err != nil {
			t.Fatal(err)
		}
	}()
	cache := NewStateMemory()
	checkpointManager, err := checkpointmanager.NewCheckpointManager(testingDir)
	require.NoError(t, err, "failed to create checkpoint manager")
	checkpointName := "pod_state_checkpoint"
	sc := &stateCheckpoint{
		cache:             cache,
		checkpointManager: checkpointManager,
		checkpointName:    checkpointName,
	}
	return sc
}

func Test_stateCheckpoint_storeState(t *testing.T) {
	sc := newTestStateCheckpoint(t)
	type args struct {
		podResourceAllocation PodResourceAllocation
	}

	tests := []struct {
		name string
		args args
	}{}
	suffix := []string{"Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "n", "u", "m", "k", "M", "G", "T", "P", "E", ""}
	factor := []string{"1", "0.1", "0.03", "10", "100", "512", "1000", "1024", "700", "10000"}
	for _, fact := range factor {
		for _, suf := range suffix {
			if (suf == "E" || suf == "Ei") && (fact == "1000" || fact == "10000") {
				// when fact is 1000 or 10000, suffix "E" or "Ei", the quantity value is overflow
				// see detail https://github.com/kubernetes/apimachinery/blob/95b78024e3feada7739b40426690b4f287933fd8/pkg/api/resource/quantity.go#L301
				continue
			}
			tests = append(tests, struct {
				name string
				args args
			}{
				name: fmt.Sprintf("resource - %s%s", fact, suf),
				args: args{
					podResourceAllocation: PodResourceAllocation{
						"pod1": {
							"container1": {
								v1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%s%s", fact, suf)),
								v1.ResourceMemory: resource.MustParse(fmt.Sprintf("%s%s", fact, suf)),
							},
						},
					},
				},
			})
		}
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := sc.cache.ClearState()
			require.NoError(t, err, "failed to clear state")

			defer func() {
				err = sc.checkpointManager.RemoveCheckpoint(sc.checkpointName)
				require.NoError(t, err, "failed to remove checkpoint")
			}()

			err = sc.cache.SetPodResourceAllocation(tt.args.podResourceAllocation)
			require.NoError(t, err, "failed to set pod resource allocation")

			err = sc.storeState()
			require.NoError(t, err, "failed to store state")

			// deep copy cache
			originCache := NewStateMemory()
			podAllocation := sc.cache.GetPodResourceAllocation()
			err = originCache.SetPodResourceAllocation(podAllocation)
			require.NoError(t, err, "failed to set pod resource allocation")

			err = sc.cache.ClearState()
			require.NoError(t, err, "failed to clear state")

			err = sc.restoreState()
			require.NoError(t, err, "failed to restore state")

			restoredCache := sc.cache
			require.Equal(t, len(originCache.GetPodResourceAllocation()), len(restoredCache.GetPodResourceAllocation()), "restored pod resource allocation is not equal to original pod resource allocation")
			for podUID, containerResourceList := range originCache.GetPodResourceAllocation() {
				require.Equal(t, len(containerResourceList), len(restoredCache.GetPodResourceAllocation()[podUID]), "restored pod resource allocation is not equal to original pod resource allocation")
				for containerName, resourceList := range containerResourceList {
					for name, quantity := range resourceList {
						if !quantity.Equal(restoredCache.GetPodResourceAllocation()[podUID][containerName][name]) {
							t.Errorf("restored pod resource allocation is not equal to original pod resource allocation")
						}
					}
				}
			}
		})
	}
}

func Test_stateCheckpoint_formatUpgraded(t *testing.T) {
	// Based on the PodResourceAllocationInfo struct, it's mostly possible that new field will be added
	// in struct PodResourceAllocationInfo, rather than in struct PodResourceAllocationInfo.AllocationEntries.
	// Emulate upgrade scenario by pretending that `ResizeStatusEntries` is a new field.
	// The checkpoint content doesn't have it and that shouldn't prevent the checkpoint from being loaded.
	sc := newTestStateCheckpoint(t)

	// prepare old checkpoint, ResizeStatusEntries is unset,
	// pretend that the old checkpoint is unaware for the field ResizeStatusEntries
	checkpointContent := `{"data":"{\"allocationEntries\":{\"pod1\":{\"container1\":{\"cpu\":\"1Ki\",\"memory\":\"1Ki\"}}}}","checksum":143149502}`
	checkpoint := &Checkpoint{}
	err := checkpoint.UnmarshalCheckpoint([]byte(checkpointContent))
	require.NoError(t, err, "failed to unmarshal checkpoint")

	err = sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	require.NoError(t, err, "failed to create old checkpoint")

	err = sc.restoreState()
	require.NoError(t, err, "failed to restore state")

	expectedPodResourceAllocationInfo := &PodResourceAllocationInfo{
		AllocationEntries: map[string]map[string]v1.ResourceList{
			"pod1": {
				"container1": {
					v1.ResourceCPU:    resource.MustParse("1Ki"),
					v1.ResourceMemory: resource.MustParse("1Ki"),
				},
			},
		},
		ResizeStatusEntries: map[string]v1.PodResizeStatus{},
	}
	actualPodResourceAllocationInfo := &PodResourceAllocationInfo{}
	actualPodResourceAllocationInfo.AllocationEntries = sc.cache.GetPodResourceAllocation()
	actualPodResourceAllocationInfo.ResizeStatusEntries = sc.cache.GetResizeStatus()
	require.NoError(t, err, "failed to get pod resource allocation info")
	require.Equal(t, expectedPodResourceAllocationInfo, actualPodResourceAllocationInfo, "pod resource allocation info is not equal")
}
