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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
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
	if plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{}}) {
		t.Errorf("Expected false")
	}
}

type fakeDiskManager struct {
	attachCalled bool
	detachCalled bool
}

func (fake *fakeDiskManager) MakeGlobalPDName(disk iscsiDisk) string {
	return "/tmp/fake_iscsi_path"
}
func (fake *fakeDiskManager) AttachDisk(disk iscsiDisk) error {
	globalPath := disk.manager.MakeGlobalPDName(disk)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return err
	}
	// Simulate the global mount so that the fakeMounter returns the
	// expected number of mounts for the attached disk.
	disk.mounter.Mount(globalPath, globalPath, disk.fsType, nil)

	fake.attachCalled = true
	return nil
}

func (fake *fakeDiskManager) DetachDisk(disk iscsiDisk, mntPath string) error {
	globalPath := disk.manager.MakeGlobalPDName(disk)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	fake.detachCalled = true
	return nil
}

func TestPlugin(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/iscsi")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
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
	fakeManager := &fakeDiskManager{}
	fakeMounter := &mount.FakeMounter{}
	builder, err := plug.(*ISCSIPlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
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
	cleaner, err := plug.(*ISCSIPlugin).newCleanerInternal("vol1", types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner: %v")
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
