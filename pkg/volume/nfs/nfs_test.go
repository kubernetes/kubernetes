/*
Copyright 2014 Google Inc. All rights reserved.

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

package nfs

import (
	"fmt"
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("fake", nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/nfs" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&api.Volume{VolumeSource: api.VolumeSource{NFS: &api.NFSVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&api.Volume{VolumeSource: api.VolumeSource{}}) {
		t.Errorf("Expected false")
	}
}

type fakeNFSMounter struct {
	FakeMounter mount.FakeMounter
}

func (fake *fakeNFSMounter) Mount(server string, source string, target string, readOnly bool) error {
	flags := 0
	if readOnly {
		flags |= mount.FlagReadOnly
	}
	fake.FakeMounter.MountPoints = append(fake.FakeMounter.MountPoints, mount.MountPoint{Device: server, Path: target, Type: "nfs", Opts: nil, Freq: 0, Pass: 0})
	return fake.FakeMounter.Mount(fmt.Sprintf("%s:%s", server, source), target, "nfs", 0, "")
}

func (fake *fakeNFSMounter) Unmount(target string) error {
	fake.FakeMounter.MountPoints = []mount.MountPoint{}
	return fake.FakeMounter.Unmount(target, 0)
}

func (fake *fakeNFSMounter) List() ([]mount.MountPoint, error) {
	list, _ := fake.FakeMounter.List()
	return list, nil
}

func (fake *fakeNFSMounter) IsMountPoint(dir string) (bool, error) {
	list, _ := fake.FakeMounter.List()
	isMount := len(list) > 0
	return isMount, nil
}

func TestPlugin(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/nfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{NFS: &api.NFSVolumeSource{"localhost", "/tmp", false}},
	}
	fake := &fakeNFSMounter{}
	builder, err := plug.(*nfsPlugin).newBuilderInternal(spec, &api.ObjectReference{UID: types.UID("poduid")}, fake)
	volumePath := builder.GetPath()
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
	}
	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~nfs/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}
	if err := builder.SetUp(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if builder.(*nfs).readOnly {
		t.Errorf("The volume source should not be read-only and it is.")
	}
	if len(fake.FakeMounter.Log) != 1 {
		t.Errorf("Mount was not called exactly one time. It was called %d times.", len(fake.FakeMounter.Log))
	} else {
		if fake.FakeMounter.Log[0].Action != mount.FakeActionMount {
			t.Errorf("Unexpected mounter action: %#v", fake.FakeMounter.Log[0])
		}
	}
	fake.FakeMounter.ResetLog()

	cleaner, err := plug.(*nfsPlugin).newCleanerInternal("vol1", types.UID("poduid"), fake)
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner: %v")
	}
	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if len(fake.FakeMounter.Log) != 1 {
		t.Errorf("Unmount was not called exactly one time. It was called %d times.", len(fake.FakeMounter.Log))
	} else {
		if fake.FakeMounter.Log[0].Action != mount.FakeActionUnmount {
			t.Errorf("Unexpected mounter action: %#v", fake.FakeMounter.Log[0])
		}
	}

	fake.FakeMounter.ResetLog()
}
