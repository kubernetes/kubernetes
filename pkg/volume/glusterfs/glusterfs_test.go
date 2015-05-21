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

package glusterfs

import (
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("fake", nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/glusterfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/glusterfs" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/glusterfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if !contains(plug.GetAccessModes(), api.ReadWriteOnce) || !contains(plug.GetAccessModes(), api.ReadOnlyMany) || !contains(plug.GetAccessModes(), api.ReadWriteMany) {
		t.Errorf("Expected three AccessModeTypes:  %s, %s, and %s", api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany)
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

func TestPlugin(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/glusterfs")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{"ep", "vol", false}},
	}
	ep := &api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "foo"}, Subsets: []api.EndpointSubset{{
		Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}}}}}
	var fcmd exec.FakeCmd
	fcmd = exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			// mount
			func() ([]byte, error) {
				return []byte{}, nil
			},
		},
	}
	fake := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.(*glusterfsPlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), ep, pod, &mount.FakeMounter{}, &fake)
	volumePath := builder.GetPath()
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
	}
	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~glusterfs/vol1" {
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
	cleaner, err := plug.(*glusterfsPlugin).newCleanerInternal("vol1", types.UID("poduid"), &mount.FakeMounter{})
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
}
