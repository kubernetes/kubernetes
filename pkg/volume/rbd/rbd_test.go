/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/rbd" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{}}) {
		t.Errorf("Expected false")
	}
}

type fakeDiskManager struct{}

func (fake *fakeDiskManager) MakeGlobalPDName(disk rbd) string {
	return "/tmp/fake_rbd_path"
}
func (fake *fakeDiskManager) AttachDisk(disk rbd) error {
	globalPath := disk.manager.MakeGlobalPDName(disk)
	err := os.MkdirAll(globalPath, 0750)
	if err != nil {
		return err
	}
	return nil
}

func (fake *fakeDiskManager) DetachDisk(disk rbd, mntPath string) error {
	globalPath := disk.manager.MakeGlobalPDName(disk)
	err := os.RemoveAll(globalPath)
	if err != nil {
		return err
	}
	return nil
}

func doTestPlugin(t *testing.T, spec *volume.Spec) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/rbd")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	builder, err := plug.(*RBDPlugin).newBuilderInternal(spec, types.UID("poduid"), &fakeDiskManager{}, &mount.FakeMounter{}, "secrets")
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
	}

	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~rbd/vol1" {
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

	cleaner, err := plug.(*RBDPlugin).newCleanerInternal("vol1", types.UID("poduid"), &fakeDiskManager{}, &mount.FakeMounter{})
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

	doTestPlugin(t, volume.NewSpecFromPersistentVolume(vol))
}
