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

package empty_dir

import (
	"os"
	"path"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
)

// The dir where volumes will be stored.
const basePath = "/tmp/fake"

// Construct an instance of a plugin, by name.
func makePluginUnderTest(t *testing.T, plugName string) volume.VolumePlugin {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost(basePath, nil, nil))

	plug, err := plugMgr.FindPluginByName(plugName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	return plug
}

func TestCanSupport(t *testing.T) {
	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir")

	if plug.Name() != "kubernetes.io/empty-dir" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{}}) {
		t.Errorf("Expected false")
	}
}

type fakeMountDetector struct {
	medium  storageMedium
	isMount bool
}

func (fake *fakeMountDetector) GetMountMedium(path string) (storageMedium, bool, error) {
	return fake.medium, fake.isMount, nil
}

func TestPlugin(t *testing.T) {
	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir")

	spec := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumDefault}},
	}
	mounter := mount.FakeMounter{}
	mountDetector := fakeMountDetector{}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.(*emptyDirPlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), pod, &mounter, &mountDetector, volume.VolumeOptions{""})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volPath := builder.GetPath()
	if volPath != path.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/vol1") {
		t.Errorf("Got unexpected path: %s", volPath)
	}

	if err := builder.SetUp(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volPath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if len(mounter.Log) != 0 {
		t.Errorf("Expected 0 mounter calls, got %#v", mounter.Log)
	}
	mounter.ResetLog()

	cleaner, err := plug.(*emptyDirPlugin).newCleanerInternal("vol1", types.UID("poduid"), &mounter, &fakeMountDetector{})
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volPath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if len(mounter.Log) != 0 {
		t.Errorf("Expected 0 mounter calls, got %#v", mounter.Log)
	}
	mounter.ResetLog()
}

func TestPluginTmpfs(t *testing.T) {
	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir")

	spec := &api.Volume{
		Name:         "vol1",
		VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumMemory}},
	}
	mounter := mount.FakeMounter{}
	mountDetector := fakeMountDetector{}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.(*emptyDirPlugin).newBuilderInternal(volume.NewSpecFromVolume(spec), pod, &mounter, &mountDetector, volume.VolumeOptions{""})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volPath := builder.GetPath()
	if volPath != path.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/vol1") {
		t.Errorf("Got unexpected path: %s", volPath)
	}

	if err := builder.SetUp(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volPath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}
	if len(mounter.Log) != 1 {
		t.Errorf("Expected 1 mounter call, got %#v", mounter.Log)
	} else {
		if mounter.Log[0].Action != mount.FakeActionMount || mounter.Log[0].FSType != "tmpfs" {
			t.Errorf("Unexpected mounter action: %#v", mounter.Log[0])
		}
	}
	mounter.ResetLog()

	cleaner, err := plug.(*emptyDirPlugin).newCleanerInternal("vol1", types.UID("poduid"), &mounter, &fakeMountDetector{mediumMemory, true})
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volPath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}
	if len(mounter.Log) != 1 {
		t.Errorf("Expected 1 mounter call, got %d (%v)", len(mounter.Log), mounter.Log)
	} else {
		if mounter.Log[0].Action != mount.FakeActionUnmount {
			t.Errorf("Unexpected mounter action: %#v", mounter.Log[0])
		}
	}
	mounter.ResetLog()
}

func TestPluginBackCompat(t *testing.T) {
	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir")

	spec := &api.Volume{
		Name: "vol1",
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.NewBuilder(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{""}, nil)
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volPath := builder.GetPath()
	if volPath != path.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/vol1") {
		t.Errorf("Got unexpected path: %s", volPath)
	}
}

func TestPluginLegacy(t *testing.T) {
	plug := makePluginUnderTest(t, "empty")

	if plug.Name() != "empty" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if plug.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}) {
		t.Errorf("Expected false")
	}

	spec := api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	if _, err := plug.(*emptyDirPlugin).newBuilderInternal(volume.NewSpecFromVolume(&spec), pod, &mount.FakeMounter{}, &fakeMountDetector{}, volume.VolumeOptions{""}); err == nil {
		t.Errorf("Expected failiure")
	}

	cleaner, err := plug.(*emptyDirPlugin).newCleanerInternal("vol1", types.UID("poduid"), &mount.FakeMounter{}, &fakeMountDetector{})
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}
}
