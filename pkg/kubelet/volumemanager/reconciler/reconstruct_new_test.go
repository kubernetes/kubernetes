/*
Copyright 2022 The Kubernetes Authors.

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

package reconciler

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

func TestReconstructVolumes(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)()

	tests := []struct {
		name                                string
		volumePaths                         []string
		expectedVolumesNeedReportedInUse    []string
		expectedVolumesNeedDevicePath       []string
		expectedVolumesFailedReconstruction []string
		verifyFunc                          func(rcInstance *reconciler, fakePlugin *volumetesting.FakeVolumePlugin) error
	}{
		{
			name: "when two pods are using same volume and both are deleted",
			volumePaths: []string{
				path.Join("pod1", "volumes", "fake-plugin", "pvc-abcdef"),
				path.Join("pod2", "volumes", "fake-plugin", "pvc-abcdef"),
			},
			expectedVolumesNeedReportedInUse:    []string{"fake-plugin/pvc-abcdef", "fake-plugin/pvc-abcdef"},
			expectedVolumesNeedDevicePath:       []string{"fake-plugin/pvc-abcdef", "fake-plugin/pvc-abcdef"},
			expectedVolumesFailedReconstruction: []string{},
			verifyFunc: func(rcInstance *reconciler, fakePlugin *volumetesting.FakeVolumePlugin) error {
				mountedPods := rcInstance.actualStateOfWorld.GetMountedVolumes()
				if len(mountedPods) != 0 {
					return fmt.Errorf("expected 0 certain pods in asw got %d", len(mountedPods))
				}
				allPods := rcInstance.actualStateOfWorld.GetAllMountedVolumes()
				if len(allPods) != 2 {
					return fmt.Errorf("expected 2 uncertain pods in asw got %d", len(allPods))
				}
				volumes := rcInstance.actualStateOfWorld.GetPossiblyMountedVolumesForPod("pod1")
				if len(volumes) != 1 {
					return fmt.Errorf("expected 1 uncertain volume in asw got %d", len(volumes))
				}
				return nil
			},
		},
		{
			name: "when reconstruction fails for a volume, volumes should be cleaned up",
			volumePaths: []string{
				path.Join("pod1", "volumes", "missing-plugin", "pvc-abcdef"),
			},
			expectedVolumesNeedReportedInUse:    []string{},
			expectedVolumesNeedDevicePath:       []string{},
			expectedVolumesFailedReconstruction: []string{"pvc-abcdef"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tmpKubeletDir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatalf("can't make a temp directory for kubeletPods: %v", err)
			}
			defer os.RemoveAll(tmpKubeletDir)

			// create kubelet pod directory
			tmpKubeletPodDir := filepath.Join(tmpKubeletDir, "pods")
			os.MkdirAll(tmpKubeletPodDir, 0755)

			mountPaths := []string{}

			// create pod and volume directories so as reconciler can find them.
			for _, volumePath := range tc.volumePaths {
				vp := filepath.Join(tmpKubeletPodDir, volumePath)
				mountPaths = append(mountPaths, vp)
				os.MkdirAll(vp, 0755)
			}

			rc, fakePlugin := getReconciler(tmpKubeletDir, t, mountPaths)
			rcInstance, _ := rc.(*reconciler)

			// Act
			rcInstance.reconstructVolumes()

			// Assert
			// Convert to []UniqueVolumeName
			expectedVolumes := make([]v1.UniqueVolumeName, len(tc.expectedVolumesNeedDevicePath))
			for i := range tc.expectedVolumesNeedDevicePath {
				expectedVolumes[i] = v1.UniqueVolumeName(tc.expectedVolumesNeedDevicePath[i])
			}
			if !reflect.DeepEqual(expectedVolumes, rcInstance.volumesNeedDevicePath) {
				t.Errorf("Expected expectedVolumesNeedDevicePath:\n%v\n got:\n%v", expectedVolumes, rcInstance.volumesNeedDevicePath)
			}

			expectedVolumes = make([]v1.UniqueVolumeName, len(tc.expectedVolumesNeedReportedInUse))
			for i := range tc.expectedVolumesNeedReportedInUse {
				expectedVolumes[i] = v1.UniqueVolumeName(tc.expectedVolumesNeedReportedInUse[i])
			}
			if !reflect.DeepEqual(expectedVolumes, rcInstance.volumesNeedReportedInUse) {
				t.Errorf("Expected volumesNeedReportedInUse:\n%v\n got:\n%v", expectedVolumes, rcInstance.volumesNeedReportedInUse)
			}

			volumesFailedReconstruction := sets.NewString()
			for _, vol := range rcInstance.volumesFailedReconstruction {
				volumesFailedReconstruction.Insert(vol.volumeSpecName)
			}
			if !reflect.DeepEqual(volumesFailedReconstruction.List(), tc.expectedVolumesFailedReconstruction) {
				t.Errorf("Expected volumesFailedReconstruction:\n%v\n got:\n%v", tc.expectedVolumesFailedReconstruction, volumesFailedReconstruction.List())
			}

			if tc.verifyFunc != nil {
				if err := tc.verifyFunc(rcInstance, fakePlugin); err != nil {
					t.Errorf("Test %s failed: %v", tc.name, err)
				}
			}
		})
	}
}

func TestCleanOrphanVolumes(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)()

	type podInfo struct {
		podName         string
		podUID          string
		outerVolumeName string
		innerVolumeName string
	}
	defaultPodInfo := podInfo{
		podName:         "pod1",
		podUID:          "pod1uid",
		outerVolumeName: "volume-name",
		innerVolumeName: "volume-name",
	}
	defaultVolume := podVolume{
		podName:        "pod1uid",
		volumeSpecName: "volume-name",
		volumePath:     "",
		pluginName:     "fake-plugin",
		volumeMode:     v1.PersistentVolumeFilesystem,
	}

	tests := []struct {
		name                        string
		podInfos                    []podInfo
		volumesFailedReconstruction []podVolume
		expectedUnmounts            int
	}{
		{
			name:                        "volume is in DSW and is not cleaned",
			podInfos:                    []podInfo{defaultPodInfo},
			volumesFailedReconstruction: []podVolume{defaultVolume},
			expectedUnmounts:            0,
		},
		{
			name:                        "volume is not in DSW and is cleaned",
			podInfos:                    []podInfo{},
			volumesFailedReconstruction: []podVolume{defaultVolume},
			expectedUnmounts:            1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange
			tmpKubeletDir, err := os.MkdirTemp("", "")
			if err != nil {
				t.Fatalf("can't make a temp directory for kubeletPods: %v", err)
			}
			defer os.RemoveAll(tmpKubeletDir)

			// create kubelet pod directory
			tmpKubeletPodDir := filepath.Join(tmpKubeletDir, "pods")
			os.MkdirAll(tmpKubeletPodDir, 0755)

			mountPaths := []string{}

			rc, fakePlugin := getReconciler(tmpKubeletDir, t, mountPaths)
			rcInstance, _ := rc.(*reconciler)
			rcInstance.volumesFailedReconstruction = tc.volumesFailedReconstruction

			for _, tpodInfo := range tc.podInfos {
				pod := getInlineFakePod(tpodInfo.podName, tpodInfo.podUID, tpodInfo.outerVolumeName, tpodInfo.innerVolumeName)
				volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
				podName := util.GetUniquePodName(pod)
				volumeName, err := rcInstance.desiredStateOfWorld.AddPodToVolume(
					podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, nil /* SELinuxContext */)
				if err != nil {
					t.Fatalf("Error adding volume %s to dsow: %v", volumeSpec.Name(), err)
				}
				rcInstance.actualStateOfWorld.MarkVolumeAsAttached(volumeName, volumeSpec, nodeName, "")
			}

			// Act
			rcInstance.cleanOrphanVolumes()

			// Assert
			if len(rcInstance.volumesFailedReconstruction) != 0 {
				t.Errorf("Expected volumesFailedReconstruction to be empty, got %+v", rcInstance.volumesFailedReconstruction)
			}
			// Unmount runs in a go routine, wait for its finish
			var lastErr error
			err = retryWithExponentialBackOff(testOperationBackOffDuration, func() (bool, error) {
				if err := verifyTearDownCalls(fakePlugin, tc.expectedUnmounts); err != nil {
					lastErr = err
					return false, nil
				}
				return true, nil
			})
			if err != nil {
				t.Errorf("Error waiting for volumes to get unmounted: %s: %s", err, lastErr)
			}
		})
	}
}

func verifyTearDownCalls(plugin *volumetesting.FakeVolumePlugin, expected int) error {
	unmounters := plugin.GetUnmounters()
	if len(unmounters) == 0 && (expected == 0) {
		return nil
	}
	actualCallCount := 0
	for _, unmounter := range unmounters {
		actualCallCount = unmounter.GetTearDownCallCount()
		if actualCallCount == expected {
			return nil
		}
	}
	return fmt.Errorf("expected TearDown calls %d, got %d", expected, actualCallCount)
}
