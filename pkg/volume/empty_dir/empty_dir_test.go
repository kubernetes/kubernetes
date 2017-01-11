// +build linux

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

package empty_dir

import (
	"os"
	"path"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

// Construct an instance of a plugin, by name.
func makePluginUnderTest(t *testing.T, plugName, basePath string) volume.VolumePlugin {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(basePath, nil, nil))

	plug, err := plugMgr.FindPluginByName(plugName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	return plug
}

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("emptydirTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir", tmpDir)

	if plug.GetPluginName() != "kubernetes.io/empty-dir" {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{}}}) {
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

func TestPluginEmptyRootContext(t *testing.T) {
	doTestPlugin(t, pluginTestConfig{
		medium:                 v1.StorageMediumDefault,
		expectedSetupMounts:    0,
		expectedTeardownMounts: 0})
}

type pluginTestConfig struct {
	medium                        v1.StorageMedium
	idempotent                    bool
	expectedSetupMounts           int
	shouldBeMountedBeforeTeardown bool
	expectedTeardownMounts        int
}

// doTestPlugin sets up a volume and tears it back down.
func doTestPlugin(t *testing.T, config pluginTestConfig) {
	basePath, err := utiltesting.MkTmpdir("emptydir_volume_test")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	defer os.RemoveAll(basePath)

	var (
		volumePath  = path.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/test-volume")
		metadataDir = path.Join(basePath, "pods/poduid/plugins/kubernetes.io~empty-dir/test-volume")

		plug       = makePluginUnderTest(t, "kubernetes.io/empty-dir", basePath)
		volumeName = "test-volume"
		spec       = &v1.Volume{
			Name:         volumeName,
			VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: config.medium}},
		}

		physicalMounter = mount.FakeMounter{}
		mountDetector   = fakeMountDetector{}
		pod             = &v1.Pod{ObjectMeta: v1.ObjectMeta{UID: types.UID("poduid")}}
	)

	if config.idempotent {
		physicalMounter.MountPoints = []mount.MountPoint{
			{
				Path: volumePath,
			},
		}
		util.SetReady(metadataDir)
	}

	mounter, err := plug.(*emptyDirPlugin).newMounterInternal(volume.NewSpecFromVolume(spec),
		pod,
		&physicalMounter,
		&mountDetector,
		volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volPath := mounter.GetPath()
	if volPath != volumePath {
		t.Errorf("Got unexpected path: %s", volPath)
	}

	if err := mounter.SetUp(nil); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}

	// Stat the directory and check the permission bits
	fileinfo, err := os.Stat(volPath)
	if !config.idempotent {
		if err != nil {
			if os.IsNotExist(err) {
				t.Errorf("SetUp() failed, volume path not created: %s", volPath)
			} else {
				t.Errorf("SetUp() failed: %v", err)
			}
		}
		if e, a := perm, fileinfo.Mode().Perm(); e != a {
			t.Errorf("Unexpected file mode for %v: expected: %v, got: %v", volPath, e, a)
		}
	} else if err == nil {
		// If this test is for idempotency and we were able
		// to stat the volume path, it's an error.
		t.Errorf("Volume directory was created unexpectedly")
	}

	// Check the number of mounts performed during setup
	if e, a := config.expectedSetupMounts, len(physicalMounter.Log); e != a {
		t.Errorf("Expected %v physicalMounter calls during setup, got %v", e, a)
	} else if config.expectedSetupMounts == 1 &&
		(physicalMounter.Log[0].Action != mount.FakeActionMount || physicalMounter.Log[0].FSType != "tmpfs") {
		t.Errorf("Unexpected physicalMounter action during setup: %#v", physicalMounter.Log[0])
	}
	physicalMounter.ResetLog()

	// Make an unmounter for the volume
	teardownMedium := mediumUnknown
	if config.medium == v1.StorageMediumMemory {
		teardownMedium = mediumMemory
	}
	unmounterMountDetector := &fakeMountDetector{medium: teardownMedium, isMount: config.shouldBeMountedBeforeTeardown}
	unmounter, err := plug.(*emptyDirPlugin).newUnmounterInternal(volumeName, types.UID("poduid"), &physicalMounter, unmounterMountDetector)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	// Tear down the volume
	if err := unmounter.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volPath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}

	// Check the number of physicalMounter calls during tardown
	if e, a := config.expectedTeardownMounts, len(physicalMounter.Log); e != a {
		t.Errorf("Expected %v physicalMounter calls during teardown, got %v", e, a)
	} else if config.expectedTeardownMounts == 1 && physicalMounter.Log[0].Action != mount.FakeActionUnmount {
		t.Errorf("Unexpected physicalMounter action during teardown: %#v", physicalMounter.Log[0])
	}
	physicalMounter.ResetLog()
}

func TestPluginBackCompat(t *testing.T) {
	basePath, err := utiltesting.MkTmpdir("emptydirTest")
	if err != nil {
		t.Fatalf("can't make a temp dir： %v", err)
	}
	defer os.RemoveAll(basePath)

	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir", basePath)

	spec := &v1.Volume{
		Name: "vol1",
	}
	pod := &v1.Pod{ObjectMeta: v1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volPath := mounter.GetPath()
	if volPath != path.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/vol1") {
		t.Errorf("Got unexpected path: %s", volPath)
	}
}

// TestMetrics tests that MetricProvider methods return sane values.
func TestMetrics(t *testing.T) {
	// Create an empty temp directory for the volume
	tmpDir, err := utiltesting.MkTmpdir("empty_dir_test")
	if err != nil {
		t.Fatalf("Can't make a tmp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir", tmpDir)

	spec := &v1.Volume{
		Name: "vol1",
	}
	pod := &v1.Pod{ObjectMeta: v1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}

	// Need to create the subdirectory
	os.MkdirAll(mounter.GetPath(), 0755)

	expectedEmptyDirUsage, err := volumetest.FindEmptyDirectoryUsageOnTmpfs()
	if err != nil {
		t.Errorf("Unexpected error finding expected empty directory usage on tmpfs: %v", err)
	}

	// TODO(pwittroc): Move this into a reusable testing utility
	metrics, err := mounter.GetMetrics()
	if err != nil {
		t.Errorf("Unexpected error when calling GetMetrics %v", err)
	}
	if e, a := expectedEmptyDirUsage.Value(), metrics.Used.Value(); e != a {
		t.Errorf("Unexpected value for empty directory; expected %v, got %v", e, a)
	}
	if metrics.Capacity.Value() <= 0 {
		t.Errorf("Expected Capacity to be greater than 0")
	}
	if metrics.Available.Value() <= 0 {
		t.Errorf("Expected Available to be greater than 0")
	}
}
