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
			"",
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
		{api.Volume{}, "", "", ""},
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
		vb, err := CreateVolumeBuilder(&tt.volume, tt.podID, tempDir)
		if tt.volume.Source == nil {
			if vb != nil {
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
		err = vb.SetUp()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		path := vb.GetPath()
		if path != tt.path {
			t.Errorf("Unexpected bind path. Expected %v, got %v", tt.path, path)
		}
		vc, err := CreateVolumeCleaner(tt.kind, tt.volume.Name, tt.podID, tempDir)
		if tt.kind == "" {
			if err != ErrUnsupportedVolumeType {
				t.Errorf("Unexpected error: %v", err)
			}
			continue
		}
		err = vc.TearDown()
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Errorf("TearDown() failed, original volume path not properly removed: %v", path)
		}
	}
}

func TestGetActiveVolumes(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "CreateVolumes")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	defer os.RemoveAll(tempDir)
	getActiveVolumesTests := []struct {
		name       string
		podID      string
		kind       string
		identifier string
	}{
		{"fakeName", "fakeID", "empty", "fakeID/fakeName"},
		{"fakeName2", "fakeID2", "empty", "fakeID2/fakeName2"},
	}
	expectedIdentifiers := []string{}
	for _, test := range getActiveVolumesTests {
		volumeDir := path.Join(tempDir, test.podID, "volumes", test.kind, test.name)
		os.MkdirAll(volumeDir, 0750)
		expectedIdentifiers = append(expectedIdentifiers, test.identifier)
	}
	volumeMap := GetCurrentVolumes(tempDir)
	for _, name := range expectedIdentifiers {
		if _, ok := volumeMap[name]; !ok {
			t.Errorf("Expected volume map entry not found: %v", name)
		}
	}
}
