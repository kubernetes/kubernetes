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
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"testing"

	api "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
)

// TestMain starting point for all tests.
// Surfaces klog flags by default to enable
// go test -v ./ --args <klog flags>
func TestMain(m *testing.M) {
	klog.InitFlags(flag.CommandLine)
	os.Exit(m.Run())
}

func makeTestPVWithMountOptions(name string, sizeGig int, driverName, volID string, mountOptions []string) *api.PersistentVolume {
	pv := makeTestPV(name, sizeGig, driverName, volID)
	pv.Spec.MountOptions = mountOptions
	return pv
}

func makeTestPV(name string, sizeGig int, driverName, volID string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse(
					fmt.Sprintf("%dGi", sizeGig),
				),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       driverName,
					VolumeHandle: volID,
					ReadOnly:     false,
				},
			},
		},
	}
}

func makeTestVol(name string, driverName string) *api.Volume {
	ro := false
	return &api.Volume{
		Name: name,
		VolumeSource: api.VolumeSource{
			CSI: &api.CSIVolumeSource{
				Driver:   driverName,
				ReadOnly: &ro,
			},
		},
	}
}

func getTestCSIDriver(name string, podInfoMount *bool, attachable *bool) *storagev1beta1.CSIDriver {
	return &storagev1beta1.CSIDriver{
		ObjectMeta: meta.ObjectMeta{
			Name: name,
		},
		Spec: storagev1beta1.CSIDriverSpec{
			PodInfoOnMount: podInfoMount,
			AttachRequired: attachable,
		},
	}
}

func TestSaveVolumeData(t *testing.T) {
	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)
	testCases := []struct {
		name       string
		data       map[string]string
		shouldFail bool
	}{
		{name: "test with data ok", data: map[string]string{"key0": "val0", "_key1": "val1", "key2": "val2"}},
		{name: "test with data ok 2 ", data: map[string]string{"_key0_": "val0", "&key1": "val1", "key2": "val2"}},
	}

	for i, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		specVolID := fmt.Sprintf("spec-volid-%d", i)
		mountDir := filepath.Join(getTargetPath(testPodUID, specVolID, plug.host), "/mount")
		if err := os.MkdirAll(mountDir, 0755); err != nil && !os.IsNotExist(err) {
			t.Errorf("failed to create dir [%s]: %v", mountDir, err)
		}

		err := saveVolumeData(path.Dir(mountDir), volDataFileName, tc.data)

		if !tc.shouldFail && err != nil {
			t.Errorf("unexpected failure: %v", err)
		}
		// did file get created
		dataDir := getTargetPath(testPodUID, specVolID, plug.host)
		file := filepath.Join(dataDir, volDataFileName)
		if _, err := os.Stat(file); err != nil {
			t.Errorf("failed to create data dir: %v", err)
		}

		// validate content
		data, err := ioutil.ReadFile(file)
		if !tc.shouldFail && err != nil {
			t.Errorf("failed to read data file: %v", err)
		}

		jsonData := new(bytes.Buffer)
		if err := json.NewEncoder(jsonData).Encode(tc.data); err != nil {
			t.Errorf("failed to encode json: %v", err)
		}
		if string(data) != jsonData.String() {
			t.Errorf("expecting encoded data %v, got %v", string(data), jsonData)
		}
	}
}
