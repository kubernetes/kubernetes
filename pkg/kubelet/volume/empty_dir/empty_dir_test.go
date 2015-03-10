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

package empty_dir

import (
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.PluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), &volume.FakeHost{"/tmp/fake", nil})

	plug, err := plugMgr.FindPluginByName("kubernetes.io/empty-dir")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/empty-dir" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&api.Volume{VolumeSource: api.VolumeSource{}}) {
		t.Errorf("Expected true")
	}
}

func TestPlugin(t *testing.T) {
	plugMgr := volume.PluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), &volume.FakeHost{"/tmp/fake", nil})

	plug, err := plugMgr.FindPluginByName("kubernetes.io/empty-dir")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
	}
	builder, err := plug.NewBuilder(spec, &api.ObjectReference{UID: types.UID("poduid")})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
	}

	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~empty-dir/vol1" {
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

	cleaner, err := plug.NewCleaner("vol1", types.UID("poduid"))
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

func TestPluginBackCompat(t *testing.T) {
	plugMgr := volume.PluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), &volume.FakeHost{"/tmp/fake", nil})

	plug, err := plugMgr.FindPluginByName("kubernetes.io/empty-dir")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
	}
	builder, err := plug.NewBuilder(spec, &api.ObjectReference{UID: types.UID("poduid")})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
	}

	path := builder.GetPath()
	if path != "/tmp/fake/pods/poduid/volumes/kubernetes.io~empty-dir/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}
}

func TestPluginLegacy(t *testing.T) {
	plugMgr := volume.PluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), &volume.FakeHost{"/tmp/fake", nil})

	plug, err := plugMgr.FindPluginByName("empty")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "empty" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if plug.CanSupport(&api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}) {
		t.Errorf("Expected false")
	}

	if _, err := plug.NewBuilder(&api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}, &api.ObjectReference{UID: types.UID("poduid")}); err == nil {
		t.Errorf("Expected failiure")
	}

	cleaner, err := plug.NewCleaner("vol1", types.UID("poduid"))
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner: %v")
	}
}
