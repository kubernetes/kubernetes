/*
Copyright 2024 The Kubernetes Authors.

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
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
)

const testCheckpoint = "pod_status_manager_state"

func newTestStateCheckpoint(t *testing.T) *stateCheckpoint {
	logger, _ := ktesting.NewTestContext(t)
	testingDir := getTestDir(t)
	cache := NewStateMemory(logger, PodResourceInfoMap{})
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

func getTestDir(t *testing.T) string {
	testingDir, err := os.MkdirTemp("", "pod_resource_allocation_state_test")
	require.NoError(t, err, "failed to create temp dir")
	t.Cleanup(func() {
		if err := os.RemoveAll(testingDir); err != nil {
			t.Fatal(err)
		}
	})
	return testingDir
}

func verifyPodResourceAllocation(t *testing.T, expected, actual *PodResourceInfoMap, msgAndArgs string) {
	for podUID, expectedPodInfo := range *expected {
		actualPodInfo, exists := (*actual)[podUID]
		require.True(t, exists, "actual state missing pod %s", podUID)
		require.NotNil(t, actualPodInfo.PodSpec, msgAndArgs)
		require.NotNil(t, expectedPodInfo.PodSpec, msgAndArgs)

		// Compare PodSpec resources
		require.True(t, apiequality.Semantic.DeepEqual(expectedPodInfo.PodSpec, actualPodInfo.PodSpec), msgAndArgs)
	}
}

func Test_stateCheckpoint_storeState(t *testing.T) {
	type args struct {
		resInfoMap PodResourceInfoMap
	}
	type testCase struct {
		name string
		args args
	}

	var tests []testCase
	suffix := []string{"Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "n", "u", "m", "k", "M", "G", "T", "P", "E", ""}
	factor := []string{"1", "0.1", "0.03", "10", "100", "512", "1000", "1024", "700", "10000"}
	for _, fact := range factor {
		for _, suf := range suffix {
			if (suf == "E" || suf == "Ei") && (fact == "1000" || fact == "10000") {
				// when fact is 1000 or 10000, suffix "E" or "Ei", the quantity value is overflow
				// see detail https://github.com/kubernetes/apimachinery/blob/95b78024e3feada7739b40426690b4f287933fd8/pkg/api/resource/quantity.go#L301
				continue
			}
			qStr := fmt.Sprintf("%s%s", fact, suf)

			// Test case 1: All fields populated
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - all fields populated", qStr),
				args: args{
					resInfoMap: PodResourceInfoMap{
						"pod1": {
							PodSpec: &v1.PodSpec{
								Containers: []v1.Container{
									{
										Name: "container1",
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse(qStr),
												v1.ResourceMemory: resource.MustParse(qStr),
											},
										},
									},
								},
								Resources: &v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse(qStr),
										v1.ResourceMemory: resource.MustParse(qStr),
									},
								},
								Volumes: []v1.Volume{
									{
										Name: "volume1",
										VolumeSource: v1.VolumeSource{
											EmptyDir: &v1.EmptyDirVolumeSource{
												SizeLimit: func() *resource.Quantity {
													q := resource.MustParse(qStr)
													return &q
												}(),
											},
										},
									},
								},
							},
						},
					},
				},
			})

			// Test case 2: Only container resources populated (pod level and volume limits are nil)
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - only container", qStr),
				args: args{
					resInfoMap: PodResourceInfoMap{
						"pod1": {
							PodSpec: &v1.PodSpec{
								Containers: []v1.Container{
									{
										Name: "container1",
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse(qStr),
												v1.ResourceMemory: resource.MustParse(qStr),
											},
										},
									},
								},
							},
						},
					},
				},
			})

			// Test case 3: Container resources and volume limits populated (pod level resources is nil)
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - container and volume limits", qStr),
				args: args{
					resInfoMap: PodResourceInfoMap{
						"pod1": {
							PodSpec: &v1.PodSpec{
								Containers: []v1.Container{
									{
										Name: "container1",
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse(qStr),
												v1.ResourceMemory: resource.MustParse(qStr),
											},
										},
									},
								},
								Volumes: []v1.Volume{
									{
										Name: "volume1",
										VolumeSource: v1.VolumeSource{
											EmptyDir: &v1.EmptyDirVolumeSource{
												SizeLimit: func() *resource.Quantity {
													q := resource.MustParse(qStr)
													return &q
												}(),
											},
										},
									},
								},
							},
						},
					},
				},
			})

			// Test case 4: Container resources and pod level resources populated (volume limits is nil)
			tests = append(tests, testCase{
				name: fmt.Sprintf("resource - %s - container and pod-level", qStr),
				args: args{
					resInfoMap: PodResourceInfoMap{
						"pod1": {
							PodSpec: &v1.PodSpec{
								Containers: []v1.Container{
									{
										Name: "container1",
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse(qStr),
												v1.ResourceMemory: resource.MustParse(qStr),
											},
										},
									},
								},
								Resources: &v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse(qStr),
										v1.ResourceMemory: resource.MustParse(qStr),
									},
								},
							},
						},
					},
				},
			})
		}
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			testDir := getTestDir(t)
			originalSC, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
			require.NoError(t, err)

			for podUID, alloc := range tt.args.resInfoMap {
				err = originalSC.SetPodResourceInfo(logger, podUID, alloc)
				require.NoError(t, err)
			}

			actual := originalSC.GetPodResourceInfoMap()
			verifyPodResourceAllocation(t, &tt.args.resInfoMap, &actual, "stored pod resource allocation is not equal to original pod resource allocation")

			newSC, err := NewStateCheckpoint(logger, testDir, testCheckpoint)
			require.NoError(t, err)

			actual = newSC.GetPodResourceInfoMap()
			verifyPodResourceAllocation(t, &tt.args.resInfoMap, &actual, "restored pod resource allocation is not equal to original pod resource allocation")

			checkpointPath := filepath.Join(testDir, testCheckpoint)
			require.FileExists(t, checkpointPath)
			require.NoError(t, os.Remove(checkpointPath)) // Remove the checkpoint file to track whether it's re-written.

			// Setting the pod allocations to the same values should not re-write the checkpoint.
			for podUID, alloc := range tt.args.resInfoMap {
				require.NoError(t, originalSC.SetPodResourceInfo(logger, podUID, alloc))
				require.NoFileExists(t, checkpointPath, "checkpoint should not be re-written")
			}

			// Setting a new value should update the checkpoint.
			require.NoError(t, originalSC.SetPodResourceInfo(logger, "foo-bar", PodResourceInfo{
				PodSpec: &v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}}},
					},
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}},
					Volumes: []v1.Volume{
						{
							Name: "volume1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									SizeLimit: func() *resource.Quantity {
										q := resource.MustParse("1")
										return &q
									}(),
								},
							},
						},
					},
				},
			}))
			require.FileExists(t, checkpointPath, "checkpoint should be re-written")
		})
	}
}

func Test_stateCheckpoint_formatUpgraded(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	sc := newTestStateCheckpoint(t)

	// prepare old checkpoint in V1 format
	const checkpointContent = `{"data":"{\"entries\":{\"pod1\":{\"ContainerResources\":{\"container1\":{\"requests\":{\"cpu\":\"1Ki\",\"memory\":\"1Ki\"}}}}}}","checksum":1178570812}`
	expectedPodResourceAllocation := PodResourceInfoMap{
		"pod1": {
			PodSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1Ki"),
								v1.ResourceMemory: resource.MustParse("1Ki"),
							},
						},
					},
				},
			},
		},
	}
	checkpoint := &Checkpoint{}
	err := checkpoint.UnmarshalCheckpoint([]byte(checkpointContent))
	require.NoError(t, err, "failed to unmarshal checkpoint")

	err = sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	require.NoError(t, err, "failed to create old checkpoint")

	actualPodResourceAllocation, _, _, err := restoreState(logger, sc.checkpointManager, sc.checkpointName)
	require.NoError(t, err, "failed to restore state")

	// Compare using verifyPodResourceAllocation to handle semantic equality of resources
	verifyPodResourceAllocation(t, &expectedPodResourceAllocation, &actualPodResourceAllocation, "migrated pod resource allocation is not equal")

	sc.cache = NewStateMemory(logger, actualPodResourceAllocation)

	actualPodResourceAllocation = sc.cache.GetPodResourceInfoMap()

	verifyPodResourceAllocation(t, &expectedPodResourceAllocation, &actualPodResourceAllocation, "cache pod resource allocation is not equal")
}
