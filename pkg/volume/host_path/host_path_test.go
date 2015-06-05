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

package host_path

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
)

func TestCanSupport(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.Name() != "kubernetes.io/host-path" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
	if !plug.CanSupport(&volume.Spec{Name: "foo", PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{}}) {
		t.Errorf("Expected false")
	}
}

func TestGetAccessModes(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	plug, err := plugMgr.FindPersistentPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if len(plug.GetAccessModes()) != 1 || plug.GetAccessModes()[0] != api.ReadWriteOnce {
		t.Errorf("Expected %s PersistentVolumeAccessMode", api.ReadWriteOnce)
	}
}

func TestRecycler(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins([]volume.VolumePlugin{&hostPathPlugin{nil, newMockRecycler}}, volume.NewFakeVolumeHost("/tmp/fake", nil, nil))

	spec := &volume.Spec{PersistentVolumeSource: api.PersistentVolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/foo"}}}
	plug, err := plugMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	recycler, err := plug.NewRecycler(spec)
	if err != nil {
		t.Error("Failed to make a new Recyler: %v", err)
	}
	if recycler.GetPath() != spec.PersistentVolumeSource.HostPath.Path {
		t.Errorf("Expected %s but got %s", spec.PersistentVolumeSource.HostPath.Path, recycler.GetPath())
	}
	if err := recycler.Recycle(); err != nil {
		t.Errorf("Mock Recycler expected to return nil but got %s", err)
	}
}

func newMockRecycler(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error) {
	return &mockRecycler{
		path: spec.PersistentVolumeSource.HostPath.Path,
	}, nil
}

type mockRecycler struct {
	path string
	host volume.VolumeHost
}

func (r *mockRecycler) GetPath() string {
	return r.path
}

func (r *mockRecycler) Recycle() error {
	// return nil means recycle passed
	return nil
}

func TestPlugin(t *testing.T) {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost("fake", nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/host-path")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{"/vol1"}},
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.NewBuilder(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{}, nil)
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	path := builder.GetPath()
	if path != "/vol1" {
		t.Errorf("Got unexpected path: %s", path)
	}

	if err := builder.SetUp(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	cleaner, err := plug.NewCleaner("vol1", types.UID("poduid"), nil)
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
}
