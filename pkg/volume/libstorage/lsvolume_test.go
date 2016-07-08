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
	"fmt"
	"os"
	"testing"

	lstypes "github.com/emccode/libstorage/api/types"
	lsutils "github.com/emccode/libstorage/api/utils"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func initPluginMgr(t *testing.T) (*volume.VolumePluginMgr, string, error) {
	plugMgr := &volume.VolumePluginMgr{}
	dir, err := utiltesting.MkTmpdir("libstorage")
	if err != nil {
		return nil, "", err
	}
	plugMgr.InitPlugins(
		ProbeVolumePlugins(lsDefaultOpts),
		volumetest.NewFakeVolumeHost(dir, nil, nil, "" /* rootContext */),
	)
	return plugMgr, dir, nil
}

func TestCanSupport(t *testing.T) {
	plugMgr, tmpDir, err := initPluginMgr(t)
	if err != nil {
		t.Error(err)
	}
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(lsPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	if plug.GetPluginName() != lsPluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}

	spec := &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				LibStorage: &api.LibStorageVolumeSource{},
			},
		},
	}
	if !plug.CanSupport(spec) {
		t.Errorf("Expected plugin to support Volume spec: %v", spec)
	}

	spec = &volume.Spec{
		PersistentVolume: &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					LibStorage: &api.LibStorageVolumeSource{},
				},
			},
		},
	}
	if !plug.CanSupport(spec) {
		t.Errorf("Expected plugin to support Persistent Volume spec: %v", spec)
	}

	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
		t.Errorf("Plugin should not support fake spec")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr, tmpDir, err := initPluginMgr(t)
	if err != nil {
		t.Error(err)
	}
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPersistentPluginByName(lsPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v by name", lsPluginName)
	}

	if len(plug.GetAccessModes()) != 1 || plug.GetAccessModes()[0] != api.ReadWriteOnce {
		t.Errorf("Expected %s PersistentVolumeAccessMode", api.ReadWriteOnce)
	}
}

func TestPlugin(t *testing.T) {
	t.Skip("Skipping for now.")
	plugMgr, tmpDir, err := initPluginMgr(t)
	if err != nil {
		t.Error(err)
	}
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(lsPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %s", lsPluginName)
	}
	lsPlug := plug.(*lsPlugin)
	lsPlug.client = newTestClient() // override client

	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	spec := &api.Volume{
		Name: "test",
		VolumeSource: api.VolumeSource{
			LibStorage: &api.LibStorageVolumeSource{
				VolumeName: "test",
			},
		},
	}

	fake := &mount.FakeMounter{}
	lsVol, err := plug.NewMounter(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if lsVol == nil {
		t.Errorf("Got a nil Mounter")
	}
	lsVol.(*lsVolume).mounter = fake

	volumePath := lsVol.GetPath()

	t.Log("mounter.GetPat() = ", volumePath)
	expectedPath := fmt.Sprintf("%s/pods/poduid/volumes/kubernetes.io~libstorage/test", tmpDir)
	if volumePath != expectedPath {
		t.Errorf("Unexpected path, expected %q, got: %q", expectedPath, volumePath)
	}

	// setup volume ahead of times
	volSize := int64(volume.RoundUpSize(1, 1024*1024*1024))
	opts := &lstypes.VolumeCreateOpts{
		Size: &volSize,
		Opts: lsutils.NewStore(),
	}

	lsPlug.client.Storage().VolumeCreate(lsPlug.ctx, "test-volume", opts)

	if err := lsVol.SetUpAt(volumePath, nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if lsVol.(*lsVolume).readOnly {
		t.Errorf("The volume source should not be read-only and it is.")
	}
	if len(fake.Log) != 1 {
		t.Errorf("Mount was not called exactly one time. It was called %d times.", len(fake.Log))
	} else {
		if fake.Log[0].Action != mount.FakeActionMount {
			t.Errorf("Unexpected mounter action: %#v", fake.Log[0])
		}
	}
	fake.ResetLog()

	unmounter, err := plug.NewUnmounter("vol1", types.UID("poduid"))
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}
	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volumePath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volumePath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if len(fake.Log) != 1 {
		t.Errorf("Unmount was not called exactly one time. It was called %d times.", len(fake.Log))
	} else {
		if fake.Log[0].Action != mount.FakeActionUnmount {
			t.Errorf("Unexpected mounter action: %#v", fake.Log[0])
		}
	}

	fake.ResetLog()
}

func TestProvisioner(t *testing.T) {
	t.Skip("Skipping for now.")
	tmpDir, err := utiltesting.MkTmpdir("lsTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(lsDefaultOpts), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, ""))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/libstorage")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	volSpec := &volume.Spec{
		PersistentVolume: &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					LibStorage: &api.LibStorageVolumeSource{},
				},
			},
		},
	}

	deleter, err := plug.(*lsPlugin).NewDeleter(volSpec)
	if err != nil {
		t.Errorf("NewDeleter() failed: %v", err)
	}

	err = deleter.Delete()
	if err != nil {
		t.Errorf("Deleter() failed: %v", err)
	}
}
