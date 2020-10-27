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

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

func TestNodeExpand(t *testing.T) {
	tests := []struct {
		name                string
		nodeExpansion       bool
		nodeStageSet        bool
		volumePhase         volume.CSIVolumePhaseType
		success             bool
		fsVolume            bool
		grpcError           error
		hasVolumeInUseError bool
		deviceStagePath     string
	}{
		{
			name:    "when node expansion is not set",
			success: false,
		},
		{
			name:            "when nodeExpansion=on, nodeStage=on, volumePhase=staged",
			nodeExpansion:   true,
			nodeStageSet:    true,
			volumePhase:     volume.CSIVolumeStaged,
			success:         true,
			fsVolume:        true,
			deviceStagePath: "/foo/bar",
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
		{
			name:                "when nodeExpansion=on, nodeStage=on, volumePhase=published has grpc volume-in-use error",
			nodeExpansion:       true,
			nodeStageSet:        true,
			volumePhase:         volume.CSIVolumePublished,
			success:             false,
			fsVolume:            true,
			grpcError:           status.Error(codes.FailedPrecondition, "volume-in-use"),
			hasVolumeInUseError: true,
		},
		{
			name:                "when nodeExpansion=on, nodeStage=on, volumePhase=published has other grpc error",
			nodeExpansion:       true,
			nodeStageSet:        true,
			volumePhase:         volume.CSIVolumePublished,
			success:             false,
			fsVolume:            true,
			grpcError:           status.Error(codes.InvalidArgument, "invalid-argument"),
			hasVolumeInUseError: false,
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
				DeviceStagePath: "/foo/bar",
				DevicePath:      "/mnt/foobar",
				CSIVolumePhase:  tc.volumePhase,
			}
			csiSource, _ := getCSISourceFromSpec(resizeOptions.VolumeSpec)
			csClient := setupClientWithExpansion(t, tc.nodeStageSet, tc.nodeExpansion)

			fakeCSIClient, _ := csClient.(*fakeCsiDriverClient)
			fakeNodeClient := fakeCSIClient.nodeClient

			if tc.grpcError != nil {
				fakeNodeClient.SetNextError(tc.grpcError)
			}

			ok, err := plug.nodeExpandWithClient(resizeOptions, csiSource, csClient, tc.fsVolume)

			if tc.hasVolumeInUseError {
				if !volumetypes.IsFailedPreconditionError(err) {
					t.Errorf("expected failed precondition error got: %v", err)
				}
			}

			// verify device staging targer path
			stagingTargetPath := fakeNodeClient.FakeNodeExpansionRequest.GetStagingTargetPath()
			if tc.deviceStagePath != "" && tc.deviceStagePath != stagingTargetPath {
				t.Errorf("For %s: expected staging path %s got %s", tc.name, tc.deviceStagePath, stagingTargetPath)
			}

			if ok != tc.success {
				if err != nil {
					t.Errorf("For %s : expected %v got %v with %v", tc.name, tc.success, ok, err)
				} else {
					t.Errorf("For %s : expected %v got %v", tc.name, tc.success, ok)
				}
			}
			// verify volume capability received by node expansion request
			if tc.success {
				capability := fakeNodeClient.FakeNodeExpansionRequest.GetVolumeCapability()
				if tc.fsVolume {
					if capability.GetMount() == nil {
						t.Errorf("For %s: expected mount accesstype got: %v", tc.name, capability)
					}
				} else {
					if capability.GetBlock() == nil {
						t.Errorf("For %s: expected block accesstype got: %v", tc.name, capability)
					}
				}
			}
		})
	}
}
