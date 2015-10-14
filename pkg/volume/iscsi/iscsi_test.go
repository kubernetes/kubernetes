/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package iscsi

import (
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/iscsi")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/iscsi" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/iscsi")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plug.GetAccessModes(), api.ReadWriteOnce) || !contains(plug.GetAccessModes(), api.ReadOnlyMany) {
		t.Errorf("Expected two AccessModeTypes:  %s and %s", api.ReadWriteOnce, api.ReadOnlyMany)
	}
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

type fakeDiskManager struct {
	attachCalled bool
	detachCalled bool
}

func (fake *fakeDiskManager) MakeGlobalPDName(disk iscsiDisk) string {
	return "/tmp/fake_iscsi_path"
}
func (fake *fakeDiskManager) AttachDisk(b iscsiDiskBuilder) error {
	globalPath := b.manager.MakeGlobalPDName(*b.iscsiDisk)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return err
	}
	// Simulate the global mount so that the fakeMounter returns the
	// expected number of mounts for the attached disk.
	b.mounter.Mount(globalPath, globalPath, b.fsType, nil)

	fake.attachCalled = true
	return nil
}

func (fake *fakeDiskManager) DetachDisk(c iscsiDiskCleaner, mntPath string) error {
	globalPath := c.manager.MakeGlobalPDName(*c.iscsiDisk)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	fake.detachCalled = true
	return nil
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/iscsi")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	fakeManager := &fakeDiskManager{}
	fakeMounter := &mount.FakeMounter{}
	builder, err := plug.(*iscsiPlugin).newBuilderInternal(spec, types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Error("Got a nil Builder")
	}

	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~iscsi/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := builder.SetUp(); err != nil {
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
	if !fakeManager.attachCalled {
		t.Errorf("Attach was not called")
	}

	fakeManager = &fakeDiskManager{}
	cleaner, err := plug.(*iscsiPlugin).newCleanerInternal("vol1", types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Error("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(path); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", path)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if !fakeManager.detachCalled {
		t.Errorf("Detach was not called")
	}
}

func TestPluginVolume(t *testing.T) {
	vol := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			ISCSI: &api.ISCSIVolumeSource{
				TargetPortal: "127.0.0.1:3260",
				IQN:          "iqn.2014-12.server:storage.target01",
				FSType:       "ext4",
				Lun:          0,
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
				ISCSI: &api.ISCSIVolumeSource{
					TargetPortal: "127.0.0.1:3260",
					IQN:          "iqn.2014-12.server:storage.target01",
					FSType:       "ext4",
					Lun:          0,
				},
			},
		},
	}
	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol, false))
}

func TestPersistentClaimReadOnlyFlag(t *testing.T) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "pvA",
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				ISCSI: &api.ISCSIVolumeSource{
					TargetPortal: "127.0.0.1:3260",
					IQN:          "iqn.2014-12.server:storage.target01",
					FSType:       "ext4",
					Lun:          0,
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

	o := testclient.NewObjects(api.Scheme, api.Scheme)
	o.Add(pv)
	o.Add(claim)
	client := &testclient.Fake{}
	client.AddReactor("*", "*", testclient.ObjectReaction(o, testapi.Default.RESTMapper()))

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", client, nil))
	plug, _ := plugMgr.FindPluginByName(iscsiPluginName)

	// readOnly bool is supplied by persistent-claim volume source when its builder creates other volumes
	spec := volume.NewSpecFromPersistentVolume(pv, true)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, _ := plug.NewBuilder(spec, pod, volume.VolumeOptions{})

	if !builder.IsReadOnly() {
		t.Errorf("Expected true for builder.IsReadOnly")
	}
}

func TestPortalBuilder(t *testing.T) {
	if portal := portalBuilder("127.0.0.1"); portal != "127.0.0.1:3260" {
		t.Errorf("wrong portal: %s", portal)
	}
	if portal := portalBuilder("127.0.0.1:3260"); portal != "127.0.0.1:3260" {
		t.Errorf("wrong portal: %s", portal)
	}
}
