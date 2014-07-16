/*
Copyright 2014 Google Inc. All rights reserved.

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

package volume

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestCreateVolumes(t *testing.T) {
	volumes := []api.Volume{
		{
			Name: "host-dir",
			Source: &api.VolumeSource{
				HostDirectory: &api.HostDirectory{"/dir/path"},
			},
		},
		{
			Name: "empty-dir",
			Source: &api.VolumeSource{
				EmptyDirectory: &api.EmptyDirectory{},
			},
		},
	}
	fakePodID := "my-id"
	expectedPaths := []string{"/dir/path", "/exports/my-id/empty-dir"}
	for i, volume := range volumes {
		extVolume, _ := CreateVolume(&volume, fakePodID)
		expectedPath := expectedPaths[i]
		path := extVolume.GetPath()
		if expectedPath != path {
			t.Errorf("Unexpected bind path. Expected %v, got %v", expectedPath, path)
		}
	}
}
