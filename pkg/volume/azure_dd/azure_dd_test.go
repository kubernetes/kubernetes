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

package azure_dd

import (
	"fmt"
	"os"
	"path"
	"testing"

	"github.com/Azure/azure-sdk-for-go/arm/compute"

	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("azure_dd")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName(azureDataDiskPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != azureDataDiskPluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{AzureDisk: &v1.AzureDiskVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}

	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{AzureDisk: &v1.AzureDiskVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

const (
	fakeDiskName = "foo"
	fakeDiskUri  = "https://azure/vhds/bar.vhd"
	fakeLun      = 2
)

type fakeAzureProvider struct {
}

func (fake *fakeAzureProvider) AttachDisk(diskName, diskUri, vmName string, lun int32, cachingMode compute.CachingTypes) error {
	if diskName != fakeDiskName || diskUri != fakeDiskUri || lun != fakeLun {
		return fmt.Errorf("wrong disk")
	}
	return nil

}

func (fake *fakeAzureProvider) DetachDiskByName(diskName, diskUri, vmName string) error {
	if diskName != fakeDiskName || diskUri != fakeDiskUri {
		return fmt.Errorf("wrong disk")
	}
	return nil
}
func (fake *fakeAzureProvider) GetDiskLun(diskName, diskUri, vmName string) (int32, error) {
	return int32(fakeLun), nil
}

func (fake *fakeAzureProvider) GetNextDiskLun(vmName string) (int32, error) {
	return fakeLun, nil
}
func (fake *fakeAzureProvider) InstanceID(name string) (string, error) {
	return "localhost", nil
}

func (fake *fakeAzureProvider) CreateVolume(name, storageAccount, storageType, location string, requestGB int) (string, string, int, error) {
	return "", "", 0, fmt.Errorf("not implemented")
}

func (fake *fakeAzureProvider) DeleteVolume(name, uri string) error {
	return fmt.Errorf("not implemented")
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("azure_ddTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName(azureDataDiskPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fs := "ext4"
	ro := false
	caching := v1.AzureDataDiskCachingNone
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			AzureDisk: &v1.AzureDiskVolumeSource{
				DiskName:    fakeDiskName,
				DataDiskURI: fakeDiskUri,
				FSType:      &fs,
				CachingMode: &caching,
				ReadOnly:    &ro,
			},
		},
	}
	mounter, err := plug.(*azureDataDiskPlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~azure-disk/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s, should be %s", path, volPath)
	}

	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	unmounter, err := plug.(*azureDataDiskPlugin).newUnmounterInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
}
