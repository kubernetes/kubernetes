/*
Copyright 2015 The Kubernetes Authors.

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

package downwardapi

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

const (
	downwardAPIDir = "..data"
	testPodUID     = types.UID("test_pod_uid")
	testNamespace  = "test_metadata_namespace"
	testName       = "test_metadata_name"
)

func TestCanSupport(t *testing.T) {
	pluginMgr := volume.VolumePluginMgr{}
	tmpDir, host := newTestHost(t, nil)
	defer os.RemoveAll(tmpDir)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.GetPluginName() != downwardAPIPluginName {
		t.Errorf("Wrong name: %s", plugin.GetPluginName())
	}
}

func TestDownwardAPI(t *testing.T) {
	labels1 := map[string]string{
		"key1": "value1",
		"key2": "value2",
	}
	labels2 := map[string]string{
		"key1": "value1",
		"key2": "value2",
		"key3": "value3",
	}
	annotations := map[string]string{
		"a1": "value1",
		"a2": "value2",
	}
	testCases := []struct {
		name           string
		files          map[string]string
		modes          map[string]int32
		podLabels      map[string]string
		podAnnotations map[string]string
		steps          []testStep
	}{
		{
			name:      "test_labels",
			files:     map[string]string{"labels": "metadata.labels"},
			podLabels: labels1,
			steps:     []testStep{{name: "labels", expectedMap: labels1}},
		},
		{
			name:           "test_annotations",
			files:          map[string]string{"annotations": "metadata.annotations"},
			podAnnotations: annotations,
			steps:          []testStep{{name: "annotations", expectedMap: annotations}},
		},
		{
			name:  "test_name",
			files: map[string]string{"name_file_name": "metadata.name"},
			steps: []testStep{{name: "name_file_name", expectedText: testName}},
		},
		{
			name:  "test_namespace",
			files: map[string]string{"namespace_file_name": "metadata.namespace"},
			steps: []testStep{{name: "namespace_file_name", expectedText: testNamespace}},
		},
		{
			name:      "test_write_twice_no_update",
			files:     map[string]string{"labels": "metadata.labels"},
			podLabels: labels1,
			steps: []testStep{
				{name: "<resetup>"},
				{name: "labels", expectedMap: labels1},
			},
		},
		{
			name:      "test_write_twice_with_update",
			files:     map[string]string{"labels": "metadata.labels"},
			podLabels: labels1,
			steps: []testStep{
				{name: "<resetup>", newLabels: labels2, linkShouldChange: true},
				{name: "labels", expectedMap: labels2},
			},
		},
		{
			name: "test_write_with_unix_path",
			files: map[string]string{
				"these/are/my/labels":        "metadata.labels",
				"these/are/your/annotations": "metadata.annotations",
			},
			podLabels:      labels1,
			podAnnotations: annotations,
			steps: []testStep{
				{name: "these/are/my/labels", expectedMap: labels1},
				{name: "these/are/your/annotations", expectedMap: annotations},
			},
		},
		{
			name:      "test_write_with_two_consecutive_slashes_in_the_path",
			files:     map[string]string{"this//labels": "metadata.labels"},
			podLabels: labels1,
			steps:     []testStep{{name: "this/labels", expectedMap: labels1}},
		},
		{
			name:  "test_default_mode",
			files: map[string]string{"name_file_name": "metadata.name"},
			steps: []testStep{{name: "name_file_name", expectedMode: 0644}},
		},
		{
			name:  "test_item_mode",
			files: map[string]string{"name_file_name": "metadata.name"},
			modes: map[string]int32{"name_file_name": 0400},
			steps: []testStep{{name: "name_file_name", expectedMode: 0400}},
		},
	}
	for _, testCase := range testCases {
		test := newDownwardAPITest(t, testCase.name, testCase.files, testCase.podLabels, testCase.podAnnotations, testCase.modes)
		for _, step := range testCase.steps {
			step.run(test)
		}
		test.tearDown()
	}
}

func newTestHost(t *testing.T, clientset clientset.Interface) (string, volume.VolumeHost) {
	tempDir, err := utiltesting.MkTmpdir("downwardApi_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return tempDir, volumetest.NewFakeVolumeHost(tempDir, clientset, empty_dir.ProbeVolumePlugins())
}

type downwardAPITest struct {
	t          *testing.T
	name       string
	plugin     volume.VolumePlugin
	pod        *v1.Pod
	mounter    volume.Mounter
	volumePath string
	rootDir    string
}

func newDownwardAPITest(t *testing.T, name string, volumeFiles, podLabels, podAnnotations map[string]string, modes map[string]int32) *downwardAPITest {
	defaultMode := int32(0644)
	var files []v1.DownwardAPIVolumeFile
	for path, fieldPath := range volumeFiles {
		file := v1.DownwardAPIVolumeFile{
			Path: path,
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: fieldPath,
			},
		}
		if mode, found := modes[path]; found {
			file.Mode = &mode
		}
		files = append(files, file)
	}
	podMeta := metav1.ObjectMeta{
		Name:        testName,
		Namespace:   testNamespace,
		Labels:      podLabels,
		Annotations: podAnnotations,
	}
	clientset := fake.NewSimpleClientset(&v1.Pod{ObjectMeta: podMeta})

	pluginMgr := volume.VolumePluginMgr{}
	rootDir, host := newTestHost(t, clientset)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	volumeSpec := &v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			DownwardAPI: &v1.DownwardAPIVolumeSource{
				DefaultMode: &defaultMode,
				Items:       files,
			},
		},
	}
	podMeta.UID = testPodUID
	pod := &v1.Pod{ObjectMeta: podMeta}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	volumePath := mounter.GetPath()

	err = mounter.SetUp(nil)
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	// downwardAPI volume should create its own empty wrapper path
	podWrapperMetadataDir := fmt.Sprintf("%v/pods/%v/plugins/kubernetes.io~empty-dir/wrapped_%v", rootDir, testPodUID, name)

	if _, err := os.Stat(podWrapperMetadataDir); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, empty-dir wrapper path was not created: %s", podWrapperMetadataDir)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	return &downwardAPITest{
		t:          t,
		plugin:     plugin,
		pod:        pod,
		mounter:    mounter,
		volumePath: volumePath,
		rootDir:    rootDir,
	}
}

func (test *downwardAPITest) tearDown() {
	unmounter, err := test.plugin.NewUnmounter(test.name, testPodUID)
	if err != nil {
		test.t.Errorf("Failed to make a new Unmounter: %v", err)
	}
	if unmounter == nil {
		test.t.Errorf("Got a nil Unmounter")
	}

	if err := unmounter.TearDown(); err != nil {
		test.t.Errorf("Expected success, got: %v", err)
	}
	if _, err := os.Stat(test.volumePath); err == nil {
		test.t.Errorf("TearDown() failed, volume path still exists: %s", test.volumePath)
	} else if !os.IsNotExist(err) {
		test.t.Errorf("SetUp() failed: %v", err)
	}
	os.RemoveAll(test.rootDir)
}

type testStep struct {
	name             string
	expectedMap      map[string]string
	expectedText     string
	expectedMode     int32
	linkShouldChange bool
	newLabels        map[string]string
}

func (s testStep) run(test *downwardAPITest) {
	switch {
	case s.newLabels != nil:
		test.pod.ObjectMeta.Labels = s.newLabels
		fallthrough
	case s.name == "<resetup>":
		verifyReSetUp(test.t, test.mounter, test.volumePath, s.linkShouldChange)
	case s.expectedMap != nil:
		verifyMapInFile(test.t, test.volumePath, s.name, s.expectedMap)
	case s.expectedText != "":
		verifyLinesInFile(test.t, test.volumePath, s.name, s.expectedText)
	case s.expectedMode != 0:
		verifyMode(test.t, test.volumePath, s.name, s.expectedMode)
	}
}

func verifyLinesInFile(t *testing.T, volumePath, filename string, expected string) {
	data, err := ioutil.ReadFile(path.Join(volumePath, filename))
	if err != nil {
		t.Errorf(err.Error())
		return
	}
	actualStr := sortLines(string(data))
	expectedStr := sortLines(expected)
	if actualStr != expectedStr {
		t.Errorf("Found `%s`, expected `%s`", actualStr, expectedStr)
	}
}

func verifyMapInFile(t *testing.T, volumePath, filename string, expected map[string]string) {
	verifyLinesInFile(t, volumePath, filename, fieldpath.FormatMap(expected))
}

func verifyMode(t *testing.T, volumePath, filename string, mode int32) {
	fileInfo, err := os.Stat(path.Join(volumePath, filename))
	if err != nil {
		t.Errorf(err.Error())
		return
	}

	actualMode := fileInfo.Mode()
	expectedMode := os.FileMode(mode)
	if actualMode != expectedMode {
		t.Errorf("Found mode `%v` expected %v", actualMode, expectedMode)
	}
}

func verifyReSetUp(t *testing.T, mounter volume.Mounter, volumePath string, shouldChange bool) {
	currentTarget, err := os.Readlink(path.Join(volumePath, downwardAPIDir))
	if err != nil {
		t.Errorf("labels file should be a link... %s\n", err.Error())
	}

	// now re-run Setup
	if err = mounter.SetUp(nil); err != nil {
		t.Errorf("Failed to re-setup volume: %v", err)
	}

	// get the link of the link
	currentTarget2, err := os.Readlink(path.Join(volumePath, downwardAPIDir))
	if err != nil {
		t.Errorf(".current should be a link... %s\n", err.Error())
	}

	switch {
	case shouldChange && currentTarget2 == currentTarget:
		t.Errorf("Got and update between the two Setup... Target link should NOT be the same\n")
	case !shouldChange && currentTarget2 != currentTarget:
		t.Errorf("No update between the two Setup... Target link should be the same %s %s\n",
			currentTarget, currentTarget2)
	}
}
