// +build !providerless

/*
Copyright 2016 The Kubernetes Authors.

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

package vsphere_volume

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/mount-utils"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/fake"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/legacy-cloud-providers/vsphere"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("vsphereVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/vsphere-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/vsphere-volume" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}

	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}

	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

type fakePDManager struct {
}

func (fake *fakePDManager) CreateVolume(v *vsphereVolumeProvisioner, selectedNode *v1.Node, selectedZone []string) (volSpec *VolumeSpec, err error) {
	volSpec = &VolumeSpec{
		Path:              "[local] test-volume-name.vmdk",
		Size:              100,
		Fstype:            "ext4",
		StoragePolicyName: "gold",
		StoragePolicyID:   "1234",
	}
	return volSpec, nil
}

func (fake *fakePDManager) DeleteVolume(vd *vsphereVolumeDeleter) error {
	if vd.volPath != "[local] test-volume-name.vmdk" {
		return fmt.Errorf("Deleter got unexpected volume path: %s", vd.volPath)
	}
	return nil
}

func TestPlugin(t *testing.T) {
	// Initial setup to test volume plugin
	tmpDir, err := utiltesting.MkTmpdir("vsphereVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/vsphere-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: "[local] test-volume-name.vmdk",
				FSType:     "ext4",
			},
		},
	}

	// Test Mounter
	fakeManager := &fakePDManager{}
	fakeMounter := mount.NewFakeMounter(nil)
	mounter, err := plug.(*vsphereVolumePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	mntPath := filepath.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~vsphere-volume/vol1")
	path := mounter.GetPath()
	if path != mntPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	// Test Unmounter
	fakeManager = &fakePDManager{}
	unmounter, err := plug.(*vsphereVolumePlugin).newUnmounterInternal("vol1", types.UID("poduid"), fakeManager, fakeMounter)
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
		t.Errorf("TearDown() failed: %v", err)
	}

	// Test Provisioner
	options := volume.VolumeOptions{
		PVC:                           volumetest.CreateTestPVC("100Mi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}
	provisioner, err := plug.(*vsphereVolumePlugin).newProvisionerInternal(options, &fakePDManager{})
	if err != nil {
		t.Errorf("newProvisionerInternal() failed: %v", err)
	}
	persistentSpec, err := provisioner.Provision(nil, nil)
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.VsphereVolume.VolumePath != "[local] test-volume-name.vmdk" {
		t.Errorf("Provision() returned unexpected path %s", persistentSpec.Spec.PersistentVolumeSource.VsphereVolume.VolumePath)
	}

	if persistentSpec.Spec.PersistentVolumeSource.VsphereVolume.StoragePolicyName != "gold" {
		t.Errorf("Provision() returned unexpected storagepolicy name %s", persistentSpec.Spec.PersistentVolumeSource.VsphereVolume.StoragePolicyName)
	}

	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 100*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*vsphereVolumePlugin).newDeleterInternal(volSpec, &fakePDManager{})
	if err != nil {
		t.Errorf("newDeleterInternal() failed: %v", err)
	}
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}

func TestUnsupportedCloudProvider(t *testing.T) {
	// Initial setup to test volume plugin
	tmpDir, err := utiltesting.MkTmpdir("vsphereVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testcases := []struct {
		name          string
		cloudProvider cloudprovider.Interface
		success       bool
	}{
		{name: "nil cloudprovider", cloudProvider: nil},
		{name: "vSphere", cloudProvider: &vsphere.VSphere{}, success: true},
		{name: "fake cloudprovider", cloudProvider: &fake.Cloud{}},
	}

	for _, tc := range testcases {
		t.Logf("test case: %v", tc.name)

		plugMgr := volume.VolumePluginMgr{}
		plugMgr.InitPlugins(ProbeVolumePlugins(), nil, /* prober */
			volumetest.NewFakeVolumeHostWithCloudProvider(t, tmpDir, nil, nil, tc.cloudProvider))

		plug, err := plugMgr.FindAttachablePluginByName("kubernetes.io/vsphere-volume")
		if err != nil {
			t.Errorf("Can't find the plugin by name")
		}

		_, err = plug.NewAttacher()
		if !tc.success && err == nil {
			t.Errorf("expected error when creating attacher due to incorrect cloud provider, but got none")
		} else if tc.success && err != nil {
			t.Errorf("expected no error when creating attacher, but got error: %v", err)
		}

		_, err = plug.NewDetacher()
		if !tc.success && err == nil {
			t.Errorf("expected error when creating detacher due to incorrect cloud provider, but got none")
		} else if tc.success && err != nil {
			t.Errorf("expected no error when creating detacher, but got error: %v", err)
		}
	}
}
