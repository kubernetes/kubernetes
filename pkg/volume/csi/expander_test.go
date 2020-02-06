/*
Copyright 2019 The Kubernetes Authors.

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

package csi

import (
	"os"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
)

func TestNodeExpand(t *testing.T) {
	tests := []struct {
		name          string
		nodeExpansion bool
		nodeStageSet  bool
		volumePhase   volume.CSIVolumePhaseType
		success       bool
		fsVolume      bool
	}{
		{
			name:    "when node expansion is not set",
			success: false,
		},
		{
			name:          "when nodeExpansion=on, nodeStage=on, volumePhase=staged",
			nodeExpansion: true,
			nodeStageSet:  true,
			volumePhase:   volume.CSIVolumeStaged,
			success:       true,
			fsVolume:      true,
		},
		{
			name:          "when nodeExpansion=on, nodeStage=off, volumePhase=staged",
			nodeExpansion: true,
			volumePhase:   volume.CSIVolumeStaged,
			success:       false,
			fsVolume:      true,
		},
		{
			name:          "when nodeExpansion=on, nodeStage=on, volumePhase=published",
			nodeExpansion: true,
			nodeStageSet:  true,
			volumePhase:   volume.CSIVolumePublished,
			success:       true,
			fsVolume:      true,
		},
		{
			name:          "when nodeExpansion=on, nodeStage=off, volumePhase=published",
			nodeExpansion: true,
			volumePhase:   volume.CSIVolumePublished,
			success:       true,
			fsVolume:      true,
		},
		{
			name:          "when nodeExpansion=on, nodeStage=off, volumePhase=published, fsVolume=false",
			nodeExpansion: true,
			volumePhase:   volume.CSIVolumePublished,
			success:       true,
			fsVolume:      false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plug, tmpDir := newTestPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "expandable", "test-vol"), false)

			newSize, _ := resource.ParseQuantity("20Gi")

			resizeOptions := volume.NodeResizeOptions{
				VolumeSpec:      spec,
				NewSize:         newSize,
				DeviceMountPath: "/foo/bar",
				DevicePath:      "/mnt/foobar",
				CSIVolumePhase:  tc.volumePhase,
			}
			csiSource, _ := getCSISourceFromSpec(resizeOptions.VolumeSpec)

			csClient := setupClientWithExpansion(t, tc.nodeStageSet, tc.nodeExpansion)

			ok, err := plug.nodeExpandWithClient(resizeOptions, csiSource, csClient, tc.fsVolume)
			if ok != tc.success {
				if err != nil {
					t.Errorf("For %s : expected %v got %v with %v", tc.name, tc.success, ok, err)
				} else {
					t.Errorf("For %s : expected %v got %v", tc.name, tc.success, ok)
				}

			}
		})
	}
}
