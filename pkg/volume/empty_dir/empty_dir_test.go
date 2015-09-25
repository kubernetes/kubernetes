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
	"io/ioutil"
	"os"
	"path"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// Construct an instance of a plugin, by name.
func makePluginUnderTest(t *testing.T, plugName, basePath string) volume.VolumePlugin {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost(basePath, nil, nil))

	plug, err := plugMgr.FindPluginByName(plugName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	return plug
}

func TestCanSupport(t *testing.T) {
	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir", "/tmp/fake")

	if plug.Name() != "kubernetes.io/empty-dir" {
		t.Errorf("Wrong name: %s", plug.Name())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}
	if plug.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{}}}) {
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

type fakeChconRequest struct {
	dir     string
	context string
}

type fakeChconRunner struct {
	requests []fakeChconRequest
}

func newFakeChconRunner() *fakeChconRunner {
	return &fakeChconRunner{}
}

func (f *fakeChconRunner) SetContext(dir, context string) error {
	f.requests = append(f.requests, fakeChconRequest{dir, context})

	return nil
}

func TestPluginEmptyRootContext(t *testing.T) {
	doTestPlugin(t, pluginTestConfig{
		medium:                 api.StorageMediumDefault,
		rootContext:            "",
		expectedChcons:         0,
		expectedSetupMounts:    0,
		expectedTeardownMounts: 0})
}

func TestPluginRootContextSet(t *testing.T) {
	if !selinuxEnabled() {
		return
	}

	doTestPlugin(t, pluginTestConfig{
		medium:                 api.StorageMediumDefault,
		rootContext:            "user:role:type:range",
		expectedSELinuxContext: "user:role:type:range",
		expectedChcons:         1,
		expectedSetupMounts:    0,
		expectedTeardownMounts: 0})
}

func TestPluginTmpfs(t *testing.T) {
	if !selinuxEnabled() {
		return
	}

	doTestPlugin(t, pluginTestConfig{
		medium:                        api.StorageMediumMemory,
		rootContext:                   "user:role:type:range",
		expectedSELinuxContext:        "user:role:type:range",
		expectedChcons:                1,
		expectedSetupMounts:           1,
		shouldBeMountedBeforeTeardown: true,
		expectedTeardownMounts:        1})
}

type pluginTestConfig struct {
	medium                        api.StorageMedium
	rootContext                   string
	SELinuxOptions                *api.SELinuxOptions
	idempotent                    bool
	expectedSELinuxContext        string
	expectedChcons                int
	expectedSetupMounts           int
	shouldBeMountedBeforeTeardown bool
	expectedTeardownMounts        int
}

// doTestPlugin sets up a volume and tears it back down.
func doTestPlugin(t *testing.T, config pluginTestConfig) {
	basePath, err := ioutil.TempDir("/tmp", "emptydir_volume_test")
	if err != nil {
		t.Fatalf("can't make a temp rootdir")
	}

	var (
		volumePath  = path.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/test-volume")
		metadataDir = path.Join(basePath, "pods/poduid/plugins/kubernetes.io~empty-dir/test-volume")

		plug       = makePluginUnderTest(t, "kubernetes.io/empty-dir", basePath)
		volumeName = "test-volume"
		spec       = &api.Volume{
			Name:         volumeName,
			VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{Medium: config.medium}},
		}

		mounter       = mount.FakeMounter{}
		mountDetector = fakeMountDetector{}
		pod           = &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
		fakeChconRnr  = &fakeChconRunner{}
	)

	// Set up the SELinux options on the pod
	if config.SELinuxOptions != nil {
		pod.Spec = api.PodSpec{
			Containers: []api.Container{
				{
					SecurityContext: &api.SecurityContext{
						SELinuxOptions: config.SELinuxOptions,
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name: volumeName,
						},
					},
				},
			},
		}
	}

	if config.idempotent {
		mounter.MountPoints = []mount.MountPoint{
			{
				Path: volumePath,
			},
		}
		util.SetReady(metadataDir)
	}

	builder, err := plug.(*emptyDirPlugin).newBuilderInternal(volume.NewSpecFromVolume(spec),
		pod,
		&mounter,
		&mountDetector,
		volume.VolumeOptions{RootContext: config.rootContext},
		fakeChconRnr)
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volPath := builder.GetPath()
	if volPath != volumePath {
		t.Errorf("Got unexpected path: %s", volPath)
	}

	if err := builder.SetUp(); err != nil {
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

	// Check the number of chcons during setup
	if e, a := config.expectedChcons, len(fakeChconRnr.requests); e != a {
		t.Errorf("Expected %v chcon calls, got %v", e, a)
	}
	if config.expectedChcons == 1 {
		if e, a := config.expectedSELinuxContext, fakeChconRnr.requests[0].context; e != a {
			t.Errorf("Unexpected chcon context argument; expected: %v, got: %v", e, a)
		}
		if e, a := volPath, fakeChconRnr.requests[0].dir; e != a {
			t.Errorf("Unexpected chcon path argument: expected: %v, got: %v", e, a)
		}
	}

	// Check the number of mounts performed during setup
	if e, a := config.expectedSetupMounts, len(mounter.Log); e != a {
		t.Errorf("Expected %v mounter calls during setup, got %v", e, a)
	} else if config.expectedSetupMounts == 1 &&
		(mounter.Log[0].Action != mount.FakeActionMount || mounter.Log[0].FSType != "tmpfs") {
		t.Errorf("Unexpected mounter action during setup: %#v", mounter.Log[0])
	}
	mounter.ResetLog()

	// Make a cleaner for the volume
	teardownMedium := mediumUnknown
	if config.medium == api.StorageMediumMemory {
		teardownMedium = mediumMemory
	}
	cleanerMountDetector := &fakeMountDetector{medium: teardownMedium, isMount: config.shouldBeMountedBeforeTeardown}
	cleaner, err := plug.(*emptyDirPlugin).newCleanerInternal(volumeName, types.UID("poduid"), &mounter, cleanerMountDetector)
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
	}

	// Tear down the volume
	if err := cleaner.TearDown(); err != nil {
		t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(volPath); err == nil {
		t.Errorf("TearDown() failed, volume path still exists: %s", volPath)
	} else if !os.IsNotExist(err) {
		t.Errorf("SetUp() failed: %v", err)
	}

	// Check the number of mounter calls during tardown
	if e, a := config.expectedTeardownMounts, len(mounter.Log); e != a {
		t.Errorf("Expected %v mounter calls during teardown, got %v", e, a)
	} else if config.expectedTeardownMounts == 1 && mounter.Log[0].Action != mount.FakeActionUnmount {
		t.Errorf("Unexpected mounter action during teardown: %#v", mounter.Log[0])
	}
	mounter.ResetLog()
}

func TestPluginBackCompat(t *testing.T) {
	basePath := "/tmp/fake"
	plug := makePluginUnderTest(t, "kubernetes.io/empty-dir", basePath)

	spec := &api.Volume{
		Name: "vol1",
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	builder, err := plug.NewBuilder(volume.NewSpecFromVolume(spec), pod, volume.VolumeOptions{RootContext: ""})
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
