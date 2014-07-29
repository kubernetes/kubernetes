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

func TestCreateVolumeBuilders(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "CreateVolumes")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(tempDir)
	createVolumesTests := []struct {
		volume api.Volume
		path   string
		podID  string
		kind   string
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
			"host",
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
			"empty",
		},
		{api.Volume{}, "", ""},
		{
			api.Volume{
				Name:   "empty-dir",
				Source: &api.VolumeSource{},
			},
			"",
			"",
			"",
		},
	}
	for _, createVolumesTest := range createVolumesTests {
		tt := createVolumesTest
		v, err := CreateVolumeBuilder(&tt.volume, tt.podID, tempDir)
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
		v, err = CreateVolumeCleaner(tt.kind)
		if tt.kind == "" {
			if err != ErrUnsupportedVolumeType {
				t.Errorf("Unexpected error: %v", err)
			}
		}
		err = v.TearDown()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}
func TestEmptySetUpAndTearDown(t *testing.T) {
	volumes := []api.Volume{
		{
			Name: "empty-dir",
			Source: &api.VolumeSource{
				EmptyDirectory: &api.EmptyDirectory{},
			},
		},
	}
	expectedPath := "/tmp/kubelet/fakeID/volumes/empty/empty-dir"
	for _, volume := range volumes {
		volumeBuilder, _ := CreateVolumeBuilder(&volume, "fakeID", "/tmp/kubelet")
		volumeBuilder.SetUp()
		if _, err := os.Stat(expectedPath); os.IsNotExist(err) {
			t.Errorf("Mount directory %v does not exist after SetUp", expectedPath)
		}
		volumeCleaner, _ := CreateVolumeCleaner("empty", expectedPath)
		volumeCleaner.TearDown()
		if _, err := os.Stat(expectedPath); !os.IsNotExist(err) {
			t.Errorf("Mount directory %v still exists after TearDown", expectedPath)
		}
	}
	os.RemoveAll("/tmp/kubelet")
}
