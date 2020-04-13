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

package kubelet

import (
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestIsHostPathDevice(t *testing.T) {
	hostPathCharDev := v1.HostPathCharDev
	testCases := []struct {
		volumeName string
		volumes    []v1.Volume
		matched    bool
	}{
		{
			volumeName: "foo",
			volumes: []v1.Volume{
				{
					Name: "bar",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "fake",
							Type: &hostPathCharDev,
						},
					},
				},
			},
			matched: false,
		},
		{
			volumeName: "foo",
			volumes: []v1.Volume{
				{
					Name: "foo",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "fake",
							Type: &hostPathCharDev,
						},
					},
				},
			},
			matched: true,
		},
		{
			volumeName: "foo",
			volumes: []v1.Volume{
				{
					Name: "foo",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
			matched: false,
		},
	}

	for idx, tc := range testCases {
		_, result := isHostPathDevice(tc.volumeName, tc.volumes)
		if result != tc.matched {
			t.Errorf("[%d] hostpath device check failed: expect %v, got %v", idx, tc.matched, result)
		}
	}
}
