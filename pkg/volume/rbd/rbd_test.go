/*
Copyright 2014 The Kubernetes Authors.

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

package rbd

import (
	"fmt"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != "kubernetes.io/rbd" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

type fakeDiskManager struct {
	tmpDir string
}

func NewFakeDiskManager() *fakeDiskManager {
	return &fakeDiskManager{
		tmpDir: utiltesting.MkTmpdirOrDie("rbd_test"),
	}
}

func (fake *fakeDiskManager) Cleanup() {
	os.RemoveAll(fake.tmpDir)
}

func (fake *fakeDiskManager) MakeGlobalPDName(disk rbd) string {
	return fake.tmpDir
}
func (fake *fakeDiskManager) AttachDisk(b rbdMounter) error {
	globalPath := b.manager.MakeGlobalPDName(*b.rbd)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakeDiskManager) DetachDisk(c rbdUnmounter, mntPath string) error {
	globalPath := c.manager.MakeGlobalPDName(*c.rbd)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakeDiskManager) CreateImage(provisioner *rbdVolumeProvisioner) (r *api.RBDVolumeSource, volumeSizeGB int, err error) {
	return nil, 0, fmt.Errorf("not implemented")
}

func (fake *fakeDiskManager) DeleteImage(deleter *rbdVolumeDeleter) error {
	return fmt.Errorf("not implemented")
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fdm := NewFakeDiskManager()
	defer fdm.Cleanup()
	mounter, err := plug.(*rbdPlugin).newMounterInternal(spec, types.UID("poduid"), fdm, &mount.FakeMounter{}, "secrets")
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Error("Got a nil Mounter")
	}

	path := mounter.GetPath()
	expectedPath := fmt.Sprintf("%s/pods/poduid/volumes/kubernetes.io~rbd/vol1", tmpDir)
	if path != expectedPath {
		t.Errorf("Unexpected path, expected %q, got: %q", expectedPath, path)
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

	unmounter, err := plug.(*rbdPlugin).newUnmounterInternal("vol1", types.UID("poduid"), fdm, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Error("Got a nil Unmounter")
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

func TestPluginVolume(t *testing.T) {
	vol := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			RBD: &api.RBDVolumeSource{
				CephMonitors: []string{"a", "b"},
				RBDImage:     "bar",
				FSType:       "ext4",
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromVolume(vol))
}
func TestPluginPersistentVolume(t *testing.T) {
	vol := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "vol1",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				RBD: &api.RBDVolumeSource{
					CephMonitors: []string{"a", "b"},
					RBDImage:     "bar",
					FSType:       "ext4",
				},
			},
		},
	}

	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("rbd_test")
	if err != nil {
		t.Fatalf("error creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				RBD: &api.RBDVolumeSource{
					CephMonitors: []string{"a", "b"},
					RBDImage:     "bar",
					FSType:       "ext4",
				},
			},
			ClaimRef: &api.ObjectReference{
				Name: "claimA",
			},
		},
	}

	claim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "claimA",
			Namespace: "nsA",
		},
		Spec: api.PersistentVolumeClaimSpec{
			VolumeName: "pvA",
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}

	client := fake.NewSimpleClientset(pv, claim)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, client, nil, "" /* rootContext */))
	plug, _ := plugMgr.FindPluginByName(rbdPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its mounter creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	mounter, _ := plug.NewMounter(spec, pod, volume.VolumeOptions{})

	if !mounter.GetAttributes().ReadOnly {
		t.Errorf("Expected true for mounter.IsReadOnly")
	}
}
