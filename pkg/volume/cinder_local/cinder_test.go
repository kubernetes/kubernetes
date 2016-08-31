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

package cinder_local

import (
	"fmt"
	"os"
	"path"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder-local")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/cinder-local" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{CinderLocal: &api.CinderLocalVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}

	if !plug.CanSupport(&volume.Spec{PersistentVolume: &api.PersistentVolume{Spec: api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{CinderLocal: &api.CinderLocalVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

type fakePDManager struct {
	// How long should AttachDisk/DetachDisk take - we need slower AttachDisk in a test.
	attachDetachDuration time.Duration
}

func getFakeDeviceName(host volume.VolumeHost, volumeID string) string {
	return path.Join(host.GetPluginDir(cinderLocalVolumePluginName), "device", volumeID)
}

// Real Cinder AttachDisk attaches a cinder volume. If it is not yet mounted,
// it mounts it it to globalPDPath.
// We create a dummy directory (="device") and bind-mount it to globalPDPath
func (fake *fakePDManager) AttachDisk(b *cinderVolumeMounter, mntPoint string) error {
	globalPath := makeGlobalPDName(b.plugin.host, b.source.VolumeID)
	fakeDeviceName := getFakeDeviceName(b.plugin.host, b.source.VolumeID)
	err := os.MkdirAll(fakeDeviceName, 0750)
	if err != nil {
		return err
	}
	// Attaching a Cinder volume can be slow...
	time.Sleep(fake.attachDetachDuration)

	// The volume is "attached", bind-mount it if it's not mounted yet.
	notmnt, err := b.mounter.IsLikelyNotMountPoint(globalPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(globalPath, 0750); err != nil {
				return err
			}
			notmnt = true
		} else {
			return err
		}
	}
	if notmnt {
		err = b.mounter.Mount(fakeDeviceName, globalPath, "", []string{"bind"})
		if err != nil {
			return err
		}
	}
	return nil
}

func (fake *fakePDManager) DetachDisk(c *cinderVolumeUnmounter, mntPoint string) error {
	globalPath := makeGlobalPDName(c.plugin.host, c.volumeID)
	fakeDeviceName := getFakeDeviceName(c.plugin.host, c.volumeID)
	// unmount the bind-mount - should be fast
	err := c.mounter.Unmount(globalPath)
	if err != nil {
		return err
	}

	// "Detach" the fake "device"
	err = os.RemoveAll(fakeDeviceName)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakePDManager) CreateVolume(c *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, secretRef string, err error) {
	return "test-volume-name", 1, "default/mysecret", nil
}

func (fake *fakePDManager) DeleteVolume(cd *cinderVolumeDeleter) error {
	if cd.pv.Spec.CinderLocal.VolumeID != "test-volume-name" {
		return fmt.Errorf("Deleter got unexpected volume name: %s", cd.pv.Spec.CinderLocal.VolumeID)
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/cinder-local")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			CinderLocal: &api.CinderLocalVolumeSource{
				CinderVolumeSource: api.CinderVolumeSource{
					VolumeID: "pd",
					FSType:   "ext4",
				},
			},
		},
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID: types.UID("poduid"),
		},
		Spec: api.PodSpec{},
	}
	mounter, err := plug.(*cinderLocalPlugin).newMounterInternal(volume.NewSpecFromVolume(spec), pod, &fakePDManager{0}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~cinder-local/vol1")
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
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", path)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	unmounter, err := plug.(*cinderLocalPlugin).newUnmounterInternal("vol1", types.UID("poduid"), &fakePDManager{0}, &mount.FakeMounter{})
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

	// Test Provisioner
	cap := resource.MustParse("100Mi")
	options := volume.VolumeOptions{
		Capacity: cap,
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
		},
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}
	provisioner, err := plug.(*cinderLocalPlugin).newProvisionerInternal(options, &fakePDManager{0})
	persistentSpec, err := provisioner.Provision()
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if persistentSpec.Spec.PersistentVolumeSource.CinderLocal.VolumeID != "test-volume-name" {
		t.Errorf("Provision() returned unexpected volume ID: %s", persistentSpec.Spec.PersistentVolumeSource.Cinder.VolumeID)
	}
	cap = persistentSpec.Spec.Capacity[api.ResourceStorage]
	size := cap.Value()
	if size != 1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: persistentSpec,
	}
	deleter, err := plug.(*cinderLocalPlugin).newDeleterInternal(volSpec, &fakePDManager{0})
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}
