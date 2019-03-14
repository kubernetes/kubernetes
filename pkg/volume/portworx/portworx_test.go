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

package portworx

import (
	"fmt"
	"os"
	"path"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	PortworxTestVolume = "portworx-test-vol"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("portworxVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/portworx-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/portworx-volume" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{PortworxVolume: &v1.PortworxVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{PortworxVolume: &v1.PortworxVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

func TestGetAccessModes(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("portworxVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/portworx-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	if !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteOnce) {
		t.Errorf("Expected to support AccessModeTypes:  %s", v1.ReadWriteOnce)
	}
	if !volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadWriteMany) {
		t.Errorf("Expected to support AccessModeTypes:  %s", v1.ReadWriteMany)
	}
	if volumetest.ContainsAccessMode(plug.GetAccessModes(), v1.ReadOnlyMany) {
		t.Errorf("Expected not to support AccessModeTypes:  %s", v1.ReadOnlyMany)
	}
}

type fakePortworxManager struct {
	attachCalled bool
	mountCalled  bool
}

func (fake *fakePortworxManager) AttachVolume(b *portworxVolumeMounter, attachOptions map[string]string) (string, error) {
	fake.attachCalled = true
	return "", nil
}

func (fake *fakePortworxManager) DetachVolume(c *portworxVolumeUnmounter) error {
	return nil
}

func (fake *fakePortworxManager) MountVolume(b *portworxVolumeMounter, mountPath string) error {
	fake.mountCalled = true
	return nil
}

func (fake *fakePortworxManager) UnmountVolume(c *portworxVolumeUnmounter, mountPath string) error {
	return nil
}

func (fake *fakePortworxManager) CreateVolume(c *portworxVolumeProvisioner) (volumeID string, volumeSizeGB int64, labels map[string]string, err error) {
	labels = make(map[string]string)
	labels["fakeportworxmanager"] = "yes"
	return PortworxTestVolume, 100, labels, nil
}

func (fake *fakePortworxManager) DeleteVolume(cd *portworxVolumeDeleter) error {
	if cd.volumeID != PortworxTestVolume {
		return fmt.Errorf("Deleter got unexpected volume name: %s", cd.volumeID)
	}
	return nil
}

func (fake *fakePortworxManager) ResizeVolume(spec *volume.Spec, newSize resource.Quantity, volumeHost volume.VolumeHost) error {
	return nil
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("portworxVolumeTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/portworx-volume")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			PortworxVolume: &v1.PortworxVolumeSource{
				VolumeID: PortworxTestVolume,
				FSType:   "ext4",
			},
		},
	}
	fakeManager := &fakePortworxManager{}
	// Test Mounter
	fakeMounter := &mount.FakeMounter{}
	mounter, err := plug.(*portworxVolumePlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~portworx-volume/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
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
	if !fakeManager.attachCalled {
		t.Errorf("Attach watch not called")
	}
	if !fakeManager.mountCalled {
		t.Errorf("Mount watch not called")
	}

	// Test Unmounter
	fakeManager = &fakePortworxManager{}
	unmounter, err := plug.(*portworxVolumePlugin).newUnmounterInternal("vol1", types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	// Test Provisioner
	options := volume.VolumeOptions{
		PVC:                           volumetest.CreateTestPVC("100Gi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}

	provisioner, err := plug.(*portworxVolumePlugin).newProvisionerInternal(options, &fakePortworxManager{})
	if err != nil {
		t.Errorf("Error creating a new provisioner:%v", err)
	}
	persistentSpec, err := provisioner.Provision(nil, nil)
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.PortworxVolume.VolumeID != PortworxTestVolume {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.PortworxVolume.VolumeID)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 100*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	if persistentSpec.Labels["fakeportworxmanager"] != "yes" {
		t.Errorf("Provision() returned unexpected labels: %v", persistentSpec.Labels)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*portworxVolumePlugin).newDeleterInternal(volSpec, &fakePortworxManager{})
	if err != nil {
		t.Errorf("Error creating a new Deleter:%v", err)
	}
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}
