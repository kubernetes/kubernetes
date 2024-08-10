//go:build linux
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

package emptydir

import (
	"fmt"
	"k8s.io/kubernetes/pkg/kubelet/util/swap"
	"os"
	"path/filepath"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/mount-utils"
)

// Construct an instance of a plugin, by name.
func makePluginUnderTest(t *testing.T, plugName, basePath string) volume.VolumePlugin {
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(t, basePath, nil, nil))

	plug, err := plugMgr.FindPluginByName(plugName)
	if err != nil {
		t.Fatal("Can't find the plugin by name")
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
	medium  v1.StorageMedium
	isMount bool
}

func (fake *fakeMountDetector) GetMountMedium(path string, requestedMedium v1.StorageMedium) (v1.StorageMedium, bool, *resource.Quantity, error) {
	return fake.medium, fake.isMount, nil, nil
}

func TestPluginEmptyRootContext(t *testing.T) {
	doTestPlugin(t, pluginTestConfig{
		volumeDirExists:        true,
		readyDirExists:         true,
		medium:                 v1.StorageMediumDefault,
		expectedSetupMounts:    0,
		expectedTeardownMounts: 0})
	doTestPlugin(t, pluginTestConfig{
		volumeDirExists:        false,
		readyDirExists:         false,
		medium:                 v1.StorageMediumDefault,
		expectedSetupMounts:    0,
		expectedTeardownMounts: 0})
	doTestPlugin(t, pluginTestConfig{
		volumeDirExists:        true,
		readyDirExists:         false,
		medium:                 v1.StorageMediumDefault,
		expectedSetupMounts:    0,
		expectedTeardownMounts: 0})
	doTestPlugin(t, pluginTestConfig{
		volumeDirExists:        false,
		readyDirExists:         true,
		medium:                 v1.StorageMediumDefault,
		expectedSetupMounts:    0,
		expectedTeardownMounts: 0})
}

func TestPluginHugetlbfs(t *testing.T) {
	testCases := map[string]struct {
		medium v1.StorageMedium
	}{
		"medium without size": {
			medium: "HugePages",
		},
		"medium with size": {
			medium: "HugePages-2Mi",
		},
	}
	for tcName, tc := range testCases {
		t.Run(tcName, func(t *testing.T) {
			doTestPlugin(t, pluginTestConfig{
				medium:                        tc.medium,
				expectedSetupMounts:           1,
				expectedTeardownMounts:        0,
				shouldBeMountedBeforeTeardown: true,
			})
		})
	}
}

type pluginTestConfig struct {
	medium v1.StorageMedium
	//volumeDirExists indicates whether volumeDir already/still exists before volume setup/teardown
	volumeDirExists bool
	//readyDirExists indicates whether readyDir already/still exists before volume setup/teardown
	readyDirExists                bool
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
		volumePath  = filepath.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/test-volume")
		metadataDir = filepath.Join(basePath, "pods/poduid/plugins/kubernetes.io~empty-dir/test-volume")

		plug       = makePluginUnderTest(t, "kubernetes.io/empty-dir", basePath)
		volumeName = "test-volume"
		spec       = &v1.Volume{
			Name:         volumeName,
			VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: config.medium}},
		}

		physicalMounter = mount.NewFakeMounter(nil)
		mountDetector   = fakeMountDetector{}
		pod             = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: types.UID("poduid"),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
							},
						},
					},
				},
			},
		}
	)

	if config.readyDirExists {
		physicalMounter.MountPoints = []mount.MountPoint{
			{
				Path: volumePath,
			},
		}
		volumeutil.SetReady(metadataDir)
	}

	mounter, err := plug.(*emptyDirPlugin).newMounterInternal(volume.NewSpecFromVolume(spec),
		pod,
		physicalMounter,
		&mountDetector)
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
	if config.volumeDirExists {
		if err := os.MkdirAll(volPath, perm); err != nil {
			t.Errorf("fail to create path: %s", volPath)
		}
	}

	// Stat the directory and check the permission bits
	testSetUp(mounter, metadataDir, volPath)

	log := physicalMounter.GetLog()
	// Check the number of mounts performed during setup
	if e, a := config.expectedSetupMounts, len(log); e != a {
		t.Errorf("Expected %v physicalMounter calls during setup, got %v", e, a)
	} else if config.expectedSetupMounts == 1 &&
		(log[0].Action != mount.FakeActionMount || (log[0].FSType != "tmpfs" && log[0].FSType != "hugetlbfs")) {
		t.Errorf("Unexpected physicalMounter action during setup: %#v", log[0])
	}
	physicalMounter.ResetLog()

	// Make an unmounter for the volume
	teardownMedium := v1.StorageMediumDefault
	if config.medium == v1.StorageMediumMemory {
		teardownMedium = v1.StorageMediumMemory
	}
	unmounterMountDetector := &fakeMountDetector{medium: teardownMedium, isMount: config.shouldBeMountedBeforeTeardown}
	unmounter, err := plug.(*emptyDirPlugin).newUnmounterInternal(volumeName, types.UID("poduid"), physicalMounter, unmounterMountDetector)
	if err != nil {
		t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		t.Errorf("Got a nil Unmounter")
	}

	if !config.readyDirExists {
		if err := os.RemoveAll(metadataDir); err != nil && !os.IsNotExist(err) {
			t.Errorf("failed to remove ready dir [%s]: %v", metadataDir, err)
		}
	}
	if !config.volumeDirExists {
		if err := os.RemoveAll(volPath); err != nil && !os.IsNotExist(err) {
			t.Errorf("failed to remove ready dir [%s]: %v", metadataDir, err)
		}
	}
	// Tear down the volume
	if err := testTearDown(unmounter, metadataDir, volPath); err != nil {
		t.Errorf("Test failed with error %v", err)
	}

	log = physicalMounter.GetLog()
	// Check the number of physicalMounter calls during tardown
	if e, a := config.expectedTeardownMounts, len(log); e != a {
		t.Errorf("Expected %v physicalMounter calls during teardown, got %v", e, a)
	} else if config.expectedTeardownMounts == 1 && log[0].Action != mount.FakeActionUnmount {
		t.Errorf("Unexpected physicalMounter action during teardown: %#v", log[0])
	}
	physicalMounter.ResetLog()
}

func testSetUp(mounter volume.Mounter, metadataDir, volPath string) error {
	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		return fmt.Errorf("expected success, got: %w", err)
	}
	// Stat the directory and check the permission bits
	if !volumeutil.IsReady(metadataDir) {
		return fmt.Errorf("SetUp() failed, ready file is not created")
	}
	fileinfo, err := os.Stat(volPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("SetUp() failed, volume path not created: %s", volPath)
		}
		return fmt.Errorf("SetUp() failed: %v", err)
	}
	if e, a := perm, fileinfo.Mode().Perm(); e != a {
		return fmt.Errorf("unexpected file mode for %v: expected: %v, got: %v", volPath, e, a)
	}
	return nil
}

func testTearDown(unmounter volume.Unmounter, metadataDir, volPath string) error {
	if err := unmounter.TearDown(); err != nil {
		return err
	}
	if volumeutil.IsReady(metadataDir) {
		return fmt.Errorf("Teardown() failed, ready file still exists")
	}
	if _, err := os.Stat(volPath); err == nil {
		return fmt.Errorf("TearDown() failed, volume path still exists: %s", volPath)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("TearDown() failed: %v", err)
	}
	return nil
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
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(volume.NewSpecFromVolume(spec), pod)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
	}

	volPath := mounter.GetPath()
	if volPath != filepath.Join(basePath, "pods/poduid/volumes/kubernetes.io~empty-dir/vol1") {
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
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	mounter, err := plug.NewMounter(volume.NewSpecFromVolume(spec), pod)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Fatalf("Got a nil Mounter")
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

func TestGetHugePagesMountOptions(t *testing.T) {
	testCases := map[string]struct {
		pod            *v1.Pod
		medium         v1.StorageMedium
		shouldFail     bool
		expectedResult string
	}{
		"ProperValues": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePages,
			shouldFail:     false,
			expectedResult: "pagesize=2Mi",
		},
		"ProperValuesAndDifferentPageSize": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
								},
							},
						},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("4Gi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePages,
			shouldFail:     false,
			expectedResult: "pagesize=1Gi",
		},
		"InitContainerAndContainerHasProperValues": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
								},
							},
						},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("4Gi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePages,
			shouldFail:     false,
			expectedResult: "pagesize=1Gi",
		},
		"InitContainerAndContainerHasDifferentPageSizes": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("2Gi"),
								},
							},
						},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("4Gi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePages,
			shouldFail:     true,
			expectedResult: "",
		},
		"ContainersWithMultiplePageSizes": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
								},
							},
						},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePages,
			shouldFail:     true,
			expectedResult: "",
		},
		"PodWithNoHugePagesRequest": {
			pod:            &v1.Pod{},
			medium:         v1.StorageMediumHugePages,
			shouldFail:     true,
			expectedResult: "",
		},
		"ProperValuesMultipleSizes": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePagesPrefix + "1Gi",
			shouldFail:     false,
			expectedResult: "pagesize=1Gi",
		},
		"InitContainerAndContainerHasProperValuesMultipleSizes": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
								},
							},
						},
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("4Gi"),
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("50Mi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePagesPrefix + "2Mi",
			shouldFail:     false,
			expectedResult: "pagesize=2Mi",
		},
		"MediumWithoutSizeMultipleSizes": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePagesPrefix,
			shouldFail:     true,
			expectedResult: "",
		},
		"IncorrectMediumFormatMultipleSizes": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
								},
							},
						},
					},
				},
			},
			medium:         "foo",
			shouldFail:     true,
			expectedResult: "",
		},
		"MediumSizeDoesntMatchResourcesMultipleSizes": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
								},
							},
						},
					},
				},
			},
			medium:         v1.StorageMediumHugePagesPrefix + "1Mi",
			shouldFail:     true,
			expectedResult: "",
		},
	}

	for testCaseName, testCase := range testCases {
		t.Run(testCaseName, func(t *testing.T) {
			value, err := getPageSizeMountOption(testCase.medium, testCase.pod)
			if testCase.shouldFail && err == nil {
				t.Errorf("%s: Unexpected success", testCaseName)
			} else if !testCase.shouldFail && err != nil {
				t.Errorf("%s: Unexpected error: %v", testCaseName, err)
			} else if testCase.expectedResult != value {
				t.Errorf("%s: Unexpected mountOptions for Pod. Expected %v, got %v", testCaseName, testCase.expectedResult, value)
			}
		})
	}
}

type testMountDetector struct {
	pageSize *resource.Quantity
	isMnt    bool
	err      error
}

func (md *testMountDetector) GetMountMedium(path string, requestedMedium v1.StorageMedium) (v1.StorageMedium, bool, *resource.Quantity, error) {
	return v1.StorageMediumHugePages, md.isMnt, md.pageSize, md.err
}

func TestSetupHugepages(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "TestSetupHugepages")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	pageSize2Mi := resource.MustParse("2Mi")

	testCases := map[string]struct {
		path       string
		ed         *emptyDir
		shouldFail bool
	}{
		"Valid: mount expected": {
			path: tmpdir,
			ed: &emptyDir{
				medium: v1.StorageMediumHugePages,
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									},
								},
							},
						},
					},
				},
				mounter: &mount.FakeMounter{},
				mountDetector: &testMountDetector{
					pageSize: &resource.Quantity{},
					isMnt:    false,
					err:      nil,
				},
			},
			shouldFail: false,
		},
		"Valid: already mounted with correct pagesize": {
			path: tmpdir,
			ed: &emptyDir{
				medium: "HugePages-2Mi",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									},
								},
							},
						},
					},
				},
				mounter: mount.NewFakeMounter([]mount.MountPoint{{Path: tmpdir, Opts: []string{"rw", "pagesize=2M", "realtime"}}}),
				mountDetector: &testMountDetector{
					pageSize: &pageSize2Mi,
					isMnt:    true,
					err:      nil,
				},
			},
			shouldFail: false,
		},
		"Valid: already mounted": {
			path: tmpdir,
			ed: &emptyDir{
				medium: "HugePages",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									},
								},
							},
						},
					},
				},
				mounter: mount.NewFakeMounter([]mount.MountPoint{{Path: tmpdir, Opts: []string{"rw", "pagesize=2M", "realtime"}}}),
				mountDetector: &testMountDetector{
					pageSize: nil,
					isMnt:    true,
					err:      nil,
				},
			},
			shouldFail: false,
		},
		"Invalid: mounter is nil": {
			path: tmpdir,
			ed: &emptyDir{
				medium: "HugePages-2Mi",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									},
								},
							},
						},
					},
				},
				mounter: nil,
			},
			shouldFail: true,
		},
		"Invalid: GetMountMedium error": {
			path: tmpdir,
			ed: &emptyDir{
				medium: "HugePages-2Mi",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									},
								},
							},
						},
					},
				},
				mounter: mount.NewFakeMounter([]mount.MountPoint{{Path: tmpdir, Opts: []string{"rw", "pagesize=2M", "realtime"}}}),
				mountDetector: &testMountDetector{
					pageSize: &pageSize2Mi,
					isMnt:    true,
					err:      fmt.Errorf("GetMountMedium error"),
				},
			},
			shouldFail: true,
		},
		"Invalid: medium and page size differ": {
			path: tmpdir,
			ed: &emptyDir{
				medium: "HugePages-1Gi",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-1Gi"): resource.MustParse("2Gi"),
									},
								},
							},
						},
					},
				},
				mounter: mount.NewFakeMounter([]mount.MountPoint{{Path: tmpdir, Opts: []string{"rw", "pagesize=2M", "realtime"}}}),
				mountDetector: &testMountDetector{
					pageSize: &pageSize2Mi,
					isMnt:    true,
					err:      nil,
				},
			},
			shouldFail: true,
		},
		"Invalid medium": {
			path: tmpdir,
			ed: &emptyDir{
				medium: "HugePages-NN",
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									},
								},
							},
						},
					},
				},
				mounter: &mount.FakeMounter{},
				mountDetector: &testMountDetector{
					pageSize: &resource.Quantity{},
					isMnt:    false,
					err:      nil,
				},
			},
			shouldFail: true,
		},
		"Invalid: setupDir fails": {
			path: "",
			ed: &emptyDir{
				medium: v1.StorageMediumHugePages,
				pod: &v1.Pod{
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceName("hugepages-2Mi"): resource.MustParse("100Mi"),
									},
								},
							},
						},
					},
				},
				mounter: &mount.FakeMounter{},
			},
			shouldFail: true,
		},
	}

	for testCaseName, testCase := range testCases {
		t.Run(testCaseName, func(t *testing.T) {
			err := testCase.ed.setupHugepages(testCase.path)
			if testCase.shouldFail && err == nil {
				t.Errorf("%s: Unexpected success", testCaseName)
			} else if !testCase.shouldFail && err != nil {
				t.Errorf("%s: Unexpected error: %v", testCaseName, err)
			}
		})
	}
}

func TestGetPageSize(t *testing.T) {
	mounter := &mount.FakeMounter{
		MountPoints: []mount.MountPoint{
			{
				Device: "/dev/sda2",
				Type:   "ext4",
				Path:   "/",
				Opts:   []string{"rw", "relatime", "errors=remount-ro"},
			},
			{
				Device: "/dev/hugepages",
				Type:   "hugetlbfs",
				Path:   "/mnt/hugepages-2Mi",
				Opts:   []string{"rw", "relatime", "pagesize=2M"},
			},
			{
				Device: "/dev/hugepages",
				Type:   "hugetlbfs",
				Path:   "/mnt/hugepages-2Mi",
				Opts:   []string{"rw", "relatime", "pagesize=2Mi"},
			},
			{
				Device: "sysfs",
				Type:   "sysfs",
				Path:   "/sys",
				Opts:   []string{"rw", "nosuid", "nodev", "noexec", "relatime"},
			},
			{
				Device: "/dev/hugepages",
				Type:   "hugetlbfs",
				Path:   "/mnt/hugepages-1Gi",
				Opts:   []string{"rw", "relatime", "pagesize=1024M"},
			},
			{
				Device: "/dev/hugepages",
				Type:   "hugetlbfs",
				Path:   "/mnt/noopt",
				Opts:   []string{"rw", "relatime"},
			},
			{
				Device: "/dev/hugepages",
				Type:   "hugetlbfs",
				Path:   "/mnt/badopt",
				Opts:   []string{"rw", "relatime", "pagesize=NN"},
			},
		},
	}

	testCases := map[string]struct {
		path           string
		mounter        mount.Interface
		expectedResult resource.Quantity
		shouldFail     bool
	}{
		"Valid: existing 2Mi mount": {
			path:           "/mnt/hugepages-2Mi",
			mounter:        mounter,
			shouldFail:     false,
			expectedResult: resource.MustParse("2Mi"),
		},
		"Valid: existing 1Gi mount": {
			path:           "/mnt/hugepages-1Gi",
			mounter:        mounter,
			shouldFail:     false,
			expectedResult: resource.MustParse("1Gi"),
		},
		"Invalid: mount point doesn't exist": {
			path:       "/mnt/nomp",
			mounter:    mounter,
			shouldFail: true,
		},
		"Invalid: no pagesize option": {
			path:       "/mnt/noopt",
			mounter:    mounter,
			shouldFail: true,
		},
		"Invalid: incorrect pagesize option": {
			path:       "/mnt/badopt",
			mounter:    mounter,
			shouldFail: true,
		},
	}

	for testCaseName, testCase := range testCases {
		t.Run(testCaseName, func(t *testing.T) {
			pageSize, err := getPageSize(testCase.path, testCase.mounter)
			if testCase.shouldFail && err == nil {
				t.Errorf("%s: Unexpected success", testCaseName)
			} else if !testCase.shouldFail && err != nil {
				t.Errorf("%s: Unexpected error: %v", testCaseName, err)
			}
			if err == nil && pageSize.Cmp(testCase.expectedResult) != 0 {
				t.Errorf("%s: Unexpected result: %s, expected: %s", testCaseName, pageSize.String(), testCase.expectedResult.String())
			}
		})
	}
}

func TestCalculateEmptyDirMemorySize(t *testing.T) {
	testCases := map[string]struct {
		pod                   *v1.Pod
		nodeAllocatableMemory resource.Quantity
		emptyDirSizeLimit     resource.Quantity
		expectedResult        resource.Quantity
		featureGateEnabled    bool
	}{
		"SizeMemoryBackedVolumesDisabled": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("memory"): resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
			},
			nodeAllocatableMemory: resource.MustParse("16Gi"),
			emptyDirSizeLimit:     resource.MustParse("1Gi"),
			expectedResult:        resource.MustParse("0"),
			featureGateEnabled:    false,
		},
		"EmptyDirLocalLimit": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceName("memory"): resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
			},
			nodeAllocatableMemory: resource.MustParse("16Gi"),
			emptyDirSizeLimit:     resource.MustParse("1Gi"),
			expectedResult:        resource.MustParse("1Gi"),
			featureGateEnabled:    true,
		},
		"PodLocalLimit": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceName("memory"): resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
			},
			nodeAllocatableMemory: resource.MustParse("16Gi"),
			emptyDirSizeLimit:     resource.MustParse("0"),
			expectedResult:        resource.MustParse("10Gi"),
			featureGateEnabled:    true,
		},
		"NodeAllocatableLimit": {
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName("memory"): resource.MustParse("10Gi"),
								},
							},
						},
					},
				},
			},
			nodeAllocatableMemory: resource.MustParse("16Gi"),
			emptyDirSizeLimit:     resource.MustParse("0"),
			expectedResult:        resource.MustParse("16Gi"),
			featureGateEnabled:    true,
		},
	}

	for testCaseName, testCase := range testCases {
		t.Run(testCaseName, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SizeMemoryBackedVolumes, testCase.featureGateEnabled)
			spec := &volume.Spec{
				Volume: &v1.Volume{
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							Medium:    v1.StorageMediumMemory,
							SizeLimit: &testCase.emptyDirSizeLimit,
						},
					},
				}}
			result := calculateEmptyDirMemorySize(&testCase.nodeAllocatableMemory, spec, testCase.pod)
			if result.Cmp(testCase.expectedResult) != 0 {
				t.Errorf("%s: Unexpected result.  Expected %v, got %v", testCaseName, testCase.expectedResult.String(), result.String())
			}
		})
	}
}

func TestTmpfsMountOptions(t *testing.T) {
	subQuantity := resource.MustParse("123Ki")

	doesStringArrayContainSubstring := func(strSlice []string, substr string) bool {
		for _, s := range strSlice {
			if strings.Contains(s, substr) {
				return true
			}
		}
		return false
	}

	testCases := map[string]struct {
		tmpfsNoswapSupported bool
		sizeLimit            resource.Quantity
	}{
		"default behavior": {},
		"tmpfs noswap is supported": {
			tmpfsNoswapSupported: true,
		},
		"size limit is non-zero": {
			sizeLimit: subQuantity,
		},
		"tmpfs noswap is supported and size limit is non-zero": {
			tmpfsNoswapSupported: true,
			sizeLimit:            subQuantity,
		},
	}

	for testCaseName, testCase := range testCases {
		t.Run(testCaseName, func(t *testing.T) {
			emptyDirObj := emptyDir{
				sizeLimit: &testCase.sizeLimit,
			}

			options := emptyDirObj.generateTmpfsMountOptions(testCase.tmpfsNoswapSupported)

			if testCase.tmpfsNoswapSupported && !doesStringArrayContainSubstring(options, swap.TmpfsNoswapOption) {
				t.Errorf("tmpfs noswap option is expected when supported. options: %v", options)
			}
			if !testCase.tmpfsNoswapSupported && doesStringArrayContainSubstring(options, swap.TmpfsNoswapOption) {
				t.Errorf("tmpfs noswap option is not expected when unsupported. options: %v", options)
			}

			if testCase.sizeLimit.IsZero() && doesStringArrayContainSubstring(options, "size=") {
				t.Errorf("size is not expected when is zero. options: %v", options)
			}
			if expectedOption := fmt.Sprintf("size=%d", testCase.sizeLimit.Value()); !testCase.sizeLimit.IsZero() && !doesStringArrayContainSubstring(options, expectedOption) {
				t.Errorf("size option is not expected when is zero. options: %v", options)
			}
		})
	}
}
