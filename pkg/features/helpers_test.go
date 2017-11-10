/*
Copyright 2015 The Kubernetes Authors.

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

package features

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestDropAlphaVolumeDevices(t *testing.T) {
	testPod := api.Pod{
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyNever,
			Containers: []api.Container{
				{
					Name:  "container1",
					Image: "testimage",
					VolumeDevices: []api.VolumeDevice{
						{
							Name:       "myvolume",
							DevicePath: "/usr/test",
						},
					},
				},
			},
			InitContainers: []api.Container{
				{
					Name:  "container1",
					Image: "testimage",
					VolumeDevices: []api.VolumeDevice{
						{
							Name:       "myvolume",
							DevicePath: "/usr/test",
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: "myvolume",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/dev/xvdc",
						},
					},
				},
			},
		},
	}

	// Enable alpha feature BlockVolume
	err1 := utilfeature.DefaultFeatureGate.Set("BlockVolume=true")
	if err1 != nil {
		t.Fatalf("Failed to enable feature gate for BlockVolume: %v", err1)
	}

	// now test dropping the fields - should not be dropped
	DropDisabledPodSpecAlphaFields(&testPod.Spec)

	// check to make sure VolumeDevices is still present
	// if featureset is set to true
	if utilfeature.DefaultFeatureGate.Enabled(BlockVolume) {
		if testPod.Spec.Containers[0].VolumeDevices == nil {
			t.Error("VolumeDevices in Container should not have been dropped based on feature-gate")
		}
		if testPod.Spec.InitContainers[0].VolumeDevices == nil {
			t.Error("VolumeDevices in Container should not have been dropped based on feature-gate")
		}
	}

	// Disable alpha feature BlockVolume
	err := utilfeature.DefaultFeatureGate.Set("BlockVolume=false")
	if err != nil {
		t.Fatalf("Failed to disable feature gate for BlockVolume: %v", err)
	}

	// now test dropping the fields
	DropDisabledPodSpecAlphaFields(&testPod.Spec)

	// check to make sure VolumeDevices is nil
	// if featureset is set to false
	if !utilfeature.DefaultFeatureGate.Enabled(BlockVolume) {
		if testPod.Spec.Containers[0].VolumeDevices != nil {
			t.Error("DropDisabledAlphaFields for Containers failed")
		}
		if testPod.Spec.InitContainers[0].VolumeDevices != nil {
			t.Error("DropDisabledAlphaFields for InitContainers failed")
		}
	}
}
