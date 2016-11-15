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

package libstorage

import (
	"os"
	"path"
	"testing"

	api "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/libstorage/lstypes"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func makePlugMgr(t *testing.T) (*volume.VolumePluginMgr, string) {
	tmpDir, err := utiltesting.MkTmpdir("libStorageTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}

	plugMgr := &volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))
	return plugMgr, tmpDir
}

func TestCanSupport(t *testing.T) {
	plugMgr, tmpDir := makePlugMgr(t)
	defer os.RemoveAll(tmpDir)
	plug, err := plugMgr.FindPluginByName(lsPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %s by name", lsPluginName)
	}
	if plug.GetPluginName() != "kubernetes.io/libstorage" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(
		&volume.Spec{
			Volume: &api.Volume{
				VolumeSource: api.VolumeSource{
					LibStorage: &api.LibStorageVolumeSource{},
				},
			},
		},
	) {
		t.Errorf("Expected true for CanSupport LibStorage VolumeSource")
	}
	if !plug.CanSupport(
		&volume.Spec{
			PersistentVolume: &api.PersistentVolume{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						LibStorage: &api.LibStorageVolumeSource{},
					},
				},
			},
		},
	) {
		t.Errorf("Expected true for CanSupport LibStorage PersistentVolumeSource")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr, tmpDir := makePlugMgr(t)
	defer os.RemoveAll(tmpDir)
	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/libstorage")
	if err != nil {
		t.Errorf("Can't find the plugin %v", "kubernetes.io/libstorage")
	}
	if !containsMode(plug.GetAccessModes(), api.ReadWriteOnce) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", api.ReadWriteOnce, api.ReadOnlyMany)
	}
}
func containsMode(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

type fakeLSMgr struct{}

func (m *fakeLSMgr) createVolume(volName string, size int64) (*lstypes.Volume, error) {
	return &lstypes.Volume{Name: "kubernetes-dynamic-pvc-0001", Size: 100}, nil
}
func (m *fakeLSMgr) attachVolume(volName string) (string, error) {
	return "/dev/sdb123", nil
}
func (m *fakeLSMgr) isAttached(volName string) (bool, error) {
	return true, nil
}
func (m *fakeLSMgr) detachVolume(volName string) error {
	return nil
}
func (m *fakeLSMgr) deleteVolume(volName string) error {
	return nil
}
func (m *fakeLSMgr) getHost() string {
	return "tcp://:12345"
}
func (m *fakeLSMgr) getService() string {
	return "ls-service"
}

func TestPlugin(t *testing.T) {
	plugMgr, tmpDir := makePlugMgr(t)
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName("kubernetes.io/libstorage")
	if err != nil {
		t.Errorf("Can't find the plugin %v", "kubernetes.io/libstorage")
	}
	lsPlug, ok := plug.(*lsPlugin)
	if !ok {
		t.Errorf("Cannot assert plugin to be type lsPlugin")
	}

	vol := &api.Volume{
		Name: "vol-0001",
		VolumeSource: api.VolumeSource{
			LibStorage: &api.LibStorageVolumeSource{
				Host:       "tcp://:1111",
				Service:    "ls-service",
				VolumeName: "vol-0001",
				FSType:     "ext4",
			},
		},
	}

	lsPlug.lsMgr = &fakeLSMgr{}
	mounter, err := lsPlug.NewMounter(
		volume.NewSpecFromVolume(vol),
		&api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}},
		volume.VolumeOptions{},
	)
	mounter.(*lsVolume).mounter = &mount.FakeMounter{}

	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volPath := path.Join(tmpDir, "pods/poduid/volumes/kubernetes.io~libstorage/vol-0001")
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

	// unmount
	unmounter, err := lsPlug.NewUnmounter("vol-0001", types.UID("poduid"))
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

	//Test Provisioner
	options := volume.VolumeOptions{
		PVC: volumetest.CreateTestPVC("100Mi", []api.PersistentVolumeAccessMode{api.ReadWriteOnce}),
		PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimDelete,
	}
	provisioner, err := lsPlug.NewProvisioner(options)
	if err != nil {
		t.Errorf("NewProvisioner failed: %v", err)
	}
	pvSpec, err := provisioner.Provision()
	if err != nil {
		t.Errorf("Provision() failed: %v", err)
	}

	if pvSpec.Spec.PersistentVolumeSource.LibStorage.VolumeName != "kubernetes-dynamic-pvc-0001" {
		t.Errorf("Provision() returned unexpected volume Name: %s",
			pvSpec.Spec.PersistentVolumeSource.LibStorage.VolumeName,
		)
	}
	cap := pvSpec.Spec.Capacity[api.ResourceStorage]
	size := cap.Value()
	if size != 100*1024*1024*1024 {
		t.Errorf("Provision() returned unexpected volume size: %v", size)
	}

	// // Test Deleter
	volSpec := &volume.Spec{
		PersistentVolume: pvSpec,
	}
	deleter, err := lsPlug.NewDeleter(volSpec)
	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}
