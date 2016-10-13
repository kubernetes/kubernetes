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

package cinder

import (
	"fmt"
	"os"
	"path"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/pkg/util/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cinderTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/cinder" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{Cinder: &v1.CinderVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}

	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{Cinder: &v1.CinderVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

type fakePDManager struct {
	// How long should AttachDisk/DetachDisk take - we need slower AttachDisk in a test.
	attachDetachDuration time.Duration
}

func getFakeDeviceName(host volume.VolumeHost, pdName string) string {
	return path.Join(host.GetPluginDir(cinderVolumePluginName), "device", pdName)
}

func (fake *fakePDManager) GetName() string {
	return "fake"
}

func (fake *fakePDManager) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	return "", nil
}

func (fake *fakePDManager) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	return map[*volume.Spec]bool{}, nil
}

func (fake *fakePDManager) WaitForAttach(spec *volume.Spec, devicePath string, timeout time.Duration) (string, error) {
	return "", nil
}

func (fake *fakePDManager) Detach(spec *volume.Spec, deviceName string, nodeName types.NodeName) error {
	return nil
}

func (fake *fakePDManager) WaitForDetach(spec *volume.Spec, devicePath string, timeout time.Duration) error {
	return nil
}

func (fake *fakePDManager) UnmountDevice(spec *volume.Spec, deviceMountPath string, mounter mount.Interface) error {
	return nil
}

func (fake *fakePDManager) CreateVolume(provisioner *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, secretRef *v1.LocalObjectReference, err error) {
	return "test-volume-name", 1, nil, nil
}

func (fake *fakePDManager) DeleteVolume(deleter *cinderVolumeDeleter) error {
	if deleter.pdName != "test-volume-name" {
		return fmt.Errorf("Deleter got unexpected volume name: %s", deleter.pdName)
	}
	return nil
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("cinderTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			Cinder: &v1.CinderVolumeSource{
				VolumeID: "pd",
				FSType:   "ext4",
			},
		},
	}
	mounter, err := newMounter(volume.NewSpecFromVolume(spec), types.UID("poduid"), plug.(*cinderPlugin), &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~cinder/vol1")
	path := mounter.GetPath()
	if path != volPath {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err = mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err = os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if _, err = os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	unmounter, err := newUnmounter("vol1", types.UID("poduid"), plug.(*cinderPlugin), &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if err = unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err = os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}

	// Test Provisioner
	options := volume.VolumeOptions{
		PVC: volumetest.CreateTestPVC("100Mi", []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
	}
	provisioner, err := newProvisioner(options, plug.(*cinderPlugin), &fakePDManager{0})
	persistentSpec, err := provisioner.Provision()
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.Cinder.VolumeID != "test-volume-name" {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.Cinder.VolumeID)
	}
	cap := persistentSpec.Spec.Capacity[v1.ResourceStorage]
	size := cap.Value()
	if size != 1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := newDeleter(volSpec, plug.(*cinderPlugin), &fakePDManager{0})
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}
