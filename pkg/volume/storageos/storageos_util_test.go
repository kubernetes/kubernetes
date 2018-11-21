/*
Copyright 2017 The Kubernetes Authors.

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

package storageos

import (
	"fmt"
	"os"

	storageostypes "github.com/storageos/go-api/types"
	"k8s.io/api/core/v1"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	"testing"
)

var testAPISecretName = "storageos-api"
var testVolName = "storageos-test-vol"
var testPVName = "storageos-test-pv"
var testNamespace = "storageos-test-namespace"
var testSize = 1
var testDesc = "testdescription"
var testPool = "testpool"
var testFSType = "ext2"
var testVolUUID = "01c43d34-89f8-83d3-422b-43536a0f25e6"

type fakeConfig struct {
	apiAddr    string
	apiUser    string
	apiPass    string
	apiVersion string
}

func (c fakeConfig) GetAPIConfig() *storageosAPIConfig {
	return &storageosAPIConfig{
		apiAddr:    "http://5.6.7.8:9999",
		apiUser:    "abc",
		apiPass:    "123",
		apiVersion: "10",
	}
}

func TestClient(t *testing.T) {
	util := storageosUtil{}
	cfg := fakeConfig{}
	err := util.NewAPI(cfg.GetAPIConfig())
	if err != nil {
		t.Fatalf("error getting api config: %v", err)
	}
	if util.api == nil {
		t.Errorf("client() unexpectedly returned nil")
	}
}

type fakeAPI struct{}

func (f fakeAPI) Volume(namespace string, ref string) (*storageostypes.Volume, error) {
	if namespace == testNamespace && ref == testVolName {
		return &storageostypes.Volume{
			ID:        "01c43d34-89f8-83d3-422b-43536a0f25e6",
			Name:      ref,
			Pool:      "default",
			Namespace: namespace,
			Size:      5,
		}, nil
	}
	return nil, fmt.Errorf("not found")
}
func (f fakeAPI) VolumeCreate(opts storageostypes.VolumeCreateOptions) (*storageostypes.Volume, error) {

	// Append a label from the api
	labels := opts.Labels
	labels["labelfromapi"] = "apilabel"

	return &storageostypes.Volume{
		ID:          testVolUUID,
		Name:        opts.Name,
		Namespace:   opts.Namespace,
		Description: opts.Description,
		Pool:        opts.Pool,
		Size:        opts.Size,
		FSType:      opts.FSType,
		Labels:      labels,
	}, nil
}
func (f fakeAPI) VolumeMount(opts storageostypes.VolumeMountOptions) error {
	return nil
}
func (f fakeAPI) VolumeUnmount(opts storageostypes.VolumeUnmountOptions) error {
	return nil
}
func (f fakeAPI) VolumeDelete(opts storageostypes.DeleteOptions) error {
	return nil
}
func (f fakeAPI) Controller(ref string) (*storageostypes.Controller, error) {
	return &storageostypes.Controller{}, nil
}

func TestCreateVolume(t *testing.T) {

	tmpDir, err := utiltesting.MkTmpdir("storageos_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	plug, _ := plugMgr.FindPluginByName("kubernetes.io/storageos")

	// Use real util with stubbed api
	util := &storageosUtil{}
	util.api = fakeAPI{}

	labels := map[string]string{
		"labelA": "valueA",
		"labelB": "valueB",
	}

	options := volume.VolumeOptions{
		PVName:                        testPVName,
		PVC:                           volumetest.CreateTestPVC(fmt.Sprintf("%dGi", testSize), []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}

	provisioner := &storageosProvisioner{
		storageosMounter: &storageosMounter{
			storageos: &storageos{
				pvName:       testPVName,
				volName:      testVolName,
				volNamespace: testNamespace,
				sizeGB:       testSize,
				pool:         testPool,
				description:  testDesc,
				fsType:       testFSType,
				labels:       labels,
				manager:      util,
				plugin:       plug.(*storageosPlugin),
			},
		},
		options: options,
	}

	vol, err := util.CreateVolume(provisioner)
	if err != nil {
		t.Errorf("CreateVolume() returned error: %v", err)
	}
	if vol == nil {
		t.Fatalf("CreateVolume() vol is empty")
	}
	if vol.ID == "" {
		t.Error("CreateVolume() vol ID is empty")
	}
	if vol.Name != testVolName {
		t.Errorf("CreateVolume() returned unexpected Name %s", vol.Name)
	}
	if vol.Namespace != testNamespace {
		t.Errorf("CreateVolume() returned unexpected Namespace %s", vol.Namespace)
	}
	if vol.Pool != testPool {
		t.Errorf("CreateVolume() returned unexpected Pool %s", vol.Pool)
	}
	if vol.FSType != testFSType {
		t.Errorf("CreateVolume() returned unexpected FSType %s", vol.FSType)
	}
	if vol.SizeGB != testSize {
		t.Errorf("CreateVolume() returned unexpected Size %d", vol.SizeGB)
	}
	if len(vol.Labels) == 0 {
		t.Error("CreateVolume() Labels are empty")
	} else {
		var val string
		var ok bool
		for k, v := range labels {
			if val, ok = vol.Labels[k]; !ok {
				t.Errorf("CreateVolume() Label %s not set", k)
			}
			if val != v {
				t.Errorf("CreateVolume() returned unexpected Label value %s", val)
			}
		}
		if val, ok = vol.Labels["labelfromapi"]; !ok {
			t.Error("CreateVolume() Label from api not set")
		}
		if val != "apilabel" {
			t.Errorf("CreateVolume() returned unexpected Label value %s", val)
		}
	}
}

func TestAttachVolume(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("storageos_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	plug, _ := plugMgr.FindPluginByName("kubernetes.io/storageos")

	// Use real util with stubbed api
	util := &storageosUtil{}
	util.api = fakeAPI{}

	mounter := &storageosMounter{
		storageos: &storageos{
			volName:      testVolName,
			volNamespace: testNamespace,
			manager:      util,
			mounter:      &mount.FakeMounter{},
			plugin:       plug.(*storageosPlugin),
		},
		deviceDir: tmpDir,
	}
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
}
