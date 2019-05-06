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
	"context"
	"testing"

	"k8s.io/kubernetes/pkg/volume"
	//"k8s.io/kubernetes/pkg/volume/csi"
)

type VolumeStatsOptions struct {
	VolumeSpec *volume.Spec

	// this just could be volumeID
	VolumeID string

	// DeviceMountPath location where device is mounted on the node. If volume type
	// is attachable - this would be global mount path otherwise
	// it would be location where volume was mounted for the pod
	DeviceMountPath string
}

func TestVolumeStats(t *testing.T) {
	spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "metrics", "test-vol"), false)
	tests := []struct {
		name           string
		volumeStatsSet bool
		volumeData     VolumeStatsOptions
		success        bool
	}{

		{
			name:           "when nodeVolumeStats=on, DevicePath=on, DeviceMountPath=on",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: true,
		},

		{
			name:           "when nodeVolumeStats=off, DevicePath=on, DeviceMountPath=on",
			volumeStatsSet: false,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: false,
		},

		{
			name:           "when nodeVolumeStats=on, DevicePath=off, DeviceMountPath=on",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "",
				DeviceMountPath: "/foo/bar",
			},
			success: false,
		},

		{
			name:           "when nodeVolumeStats=on, DevicePath=on, DeviceMountPath=off",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "",
			},
			success: false,
		},
		{
			name:           "when nodeVolumeStats=on, DevicePath=on, DeviceMountPath=off",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "",
				DeviceMountPath: "",
			},
			success: false,
		},
	}
	for _, tc := range tests {

		ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
		defer cancel()

		csiSource, _ := getCSISourceFromSpec(tc.volumeData.VolumeSpec)

		csClient := setupClientWithVolumeStats(t, tc.volumeStatsSet)

		_, err := csClient.NodeGetVolumeStats(ctx, csiSource.VolumeHandle, tc.volumeData.DeviceMountPath)

		if err != nil && tc.success {
			t.Errorf("For %s : expected %v got %v", tc.name, tc.success, err)
		}

	}

}
