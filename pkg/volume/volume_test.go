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
	"io/ioutil"
	"os"
	"path"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestCreateVolumes(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "CreateVolumes")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(tempDir)
	createVolumesTests := []struct {
		volume api.Volume
		path   string
		podID  string
	}{
		{
			api.Volume{
				Name: "host-dir",
				Source: &api.VolumeSource{
					HostDirectory: &api.HostDirectory{"/dir/path"},
				},
			},
			"/dir/path",
			"my-id",
		},
		{
			api.Volume{
				Name: "empty-dir",
				Source: &api.VolumeSource{
					EmptyDirectory: &api.EmptyDirectory{},
				},
			},
			path.Join(tempDir, "/my-id/volumes/empty/empty-dir"),
			"my-id",
		},
		{api.Volume{}, "", ""},
		{
			api.Volume{
				Name:   "empty-dir",
				Source: &api.VolumeSource{},
			},
			"",
			"",
		},
	}
	for _, createVolumesTest := range createVolumesTests {
		tt := createVolumesTest
		v, err := CreateVolume(&tt.volume, tt.podID, tempDir)
		if tt.volume.Source == nil {
			if v != nil {
				t.Errorf("Expected volume to be nil")
			}
			continue
		}
		if tt.volume.Source.HostDirectory == nil && tt.volume.Source.EmptyDirectory == nil {
			if err != ErrUnsupportedVolumeType {
				t.Errorf("Unexpected error: %v", err)
			}
			continue
		}
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		err = v.SetUp()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		path := v.GetPath()
		if path != tt.path {
			t.Errorf("Unexpected bind path. Expected %v, got %v", tt.path, path)
		}
		err = v.TearDown()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}
