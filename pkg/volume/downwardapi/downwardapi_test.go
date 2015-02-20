/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
)

const basePath = "/tmp/fake"

func formatMap(m map[string]string) string {
	var l string
	for key, value := range m {
		l += key + "=" + fmt.Sprintf("%q", value) + "\n"
	}
	return l
}

func newTestHost(t *testing.T, client client.Interface) volume.VolumeHost {
	tempDir, err := ioutil.TempDir(basePath, "downwardApi_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}
	return volume.NewFakeVolumeHost(tempDir, client, empty_dir.ProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, nil))

	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.Name() != downwardAPIPluginName {
		t.Errorf("Wrong name: %s", plugin.Name())
	}
}

func CleanEverything(plugin volume.VolumePlugin, testVolumeName, volumePath string, testPodUID types.UID, t *testing.T) {
	cleaner, err := plugin.NewCleaner(testVolumeName, testPodUID, mount.New())
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner")
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

func TestLabels(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_labels"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	labels := map[string]string{
		"key1": "value1",
		"key2": "value2"}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Labels:    labels,
		},
	})
	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "labels", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.labels"}}}},
		},
	}
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Labels: labels}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})

	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()

	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "labels"))
	if err != nil {
		t.Errorf(err.Error())
	}
	if sortLines(string(data)) != sortLines(formatMap(labels)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(labels))
	}

	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)
}

func TestAnnotations(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_annotations"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	annotations := map[string]string{
		"a1": "value1",
		"a2": "value2"}

	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "annotations", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.annotations"}}}},
		},
	}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:        testName,
			Namespace:   testNamespace,
			Annotations: annotations,
		},
	})

	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Annotations: annotations}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()

	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "annotations"))
	if err != nil {
		t.Errorf(err.Error())
	}

	if sortLines(string(data)) != sortLines(formatMap(annotations)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(annotations))
	}
	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)

}

func TestName(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_name"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "name_file_name", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.name"}}}},
		},
	}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
		},
	})

	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Name: testName}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()

	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "name_file_name"))
	if err != nil {
		t.Errorf(err.Error())
	}

	if string(data) != testName {
		t.Errorf("Found `%s` expected %s", string(data), testName)
	}

	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)

}

func TestNamespace(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_namespace"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "namespace_file_name", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.namespace"}}}},
		},
	}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
		},
	})

	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Namespace: testNamespace}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()

	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "namespace_file_name"))
	if err != nil {
		t.Errorf(err.Error())
	}
	if string(data) != testNamespace {
		t.Errorf("Found `%s` expected %s", string(data), testNamespace)
	}

	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)

}

func TestWriteTwiceNoUpdate(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_write_twice_no_update"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	labels := map[string]string{
		"key1": "value1",
		"key2": "value2"}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Labels:    labels,
		},
	})
	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "labels", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.labels"}}}},
		},
	}
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Labels: labels}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})

	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()
	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	// get the link of the link
	var currentTarget string
	if currentTarget, err = os.Readlink(path.Join(volumePath, downwardAPIDir)); err != nil {
		t.Errorf(".current should be a link... %s\n", err.Error())
	}

	err = builder.SetUp() // now re-run Setup
	if err != nil {
		t.Errorf("Failed to re-setup volume: %v", err)
	}

	// get the link of the link
	var currentTarget2 string
	if currentTarget2, err = os.Readlink(path.Join(volumePath, downwardAPIDir)); err != nil {
		t.Errorf(".current should be a link... %s\n", err.Error())
	}

	if currentTarget2 != currentTarget {
		t.Errorf("No update between the two Setup... Target link should be the same %s %s\n", currentTarget, currentTarget2)
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "labels"))
	if err != nil {
		t.Errorf(err.Error())
	}

	if sortLines(string(data)) != sortLines(formatMap(labels)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(labels))
	}
	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)

}

func TestWriteTwiceWithUpdate(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_write_twice_with_update"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	labels := map[string]string{
		"key1": "value1",
		"key2": "value2"}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Labels:    labels,
		},
	})
	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "labels", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.labels"}}}},
		},
	}
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Labels: labels}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})

	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()
	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	var currentTarget string
	if currentTarget, err = os.Readlink(path.Join(volumePath, downwardAPIDir)); err != nil {
		t.Errorf("labels file should be a link... %s\n", err.Error())
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "labels"))
	if err != nil {
		t.Errorf(err.Error())
	}

	if sortLines(string(data)) != sortLines(formatMap(labels)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(labels))
	}

	newLabels := map[string]string{
		"key1": "value1",
		"key2": "value2",
		"key3": "value3"}

	// Now update the labels
	pod.ObjectMeta.Labels = newLabels
	err = builder.SetUp() // now re-run Setup
	if err != nil {
		t.Errorf("Failed to re-setup volume: %v", err)
	}

	// get the link of the link
	var currentTarget2 string
	if currentTarget2, err = os.Readlink(path.Join(volumePath, downwardAPIDir)); err != nil {
		t.Errorf(".current should be a link... %s\n", err.Error())
	}

	if currentTarget2 == currentTarget {
		t.Errorf("Got and update between the two Setup... Target link should NOT be the same\n")
	}

	data, err = ioutil.ReadFile(path.Join(volumePath, "labels"))
	if err != nil {
		t.Errorf(err.Error())
	}

	if sortLines(string(data)) != sortLines(formatMap(newLabels)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(newLabels))
	}
	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)
}

func TestWriteWithUnixPath(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_write_with_unix_path"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	labels := map[string]string{
		"key1": "value1",
		"key2": "value2",
		"key3": "value3\n"}

	annotations := map[string]string{
		"a1": "value1",
		"a2": "value2"}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Labels:    labels,
		},
	})

	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "this/is/mine/labels", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.labels"}},
					{Path: "this/is/yours/annotations", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.annotations"}},
				}}},
	}
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Labels: labels, Annotations: annotations}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})

	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()
	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "this/is/mine/labels"))
	if err != nil {
		t.Errorf(err.Error())
	}

	if sortLines(string(data)) != sortLines(formatMap(labels)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(labels))
	}

	data, err = ioutil.ReadFile(path.Join(volumePath, "this/is/yours/annotations"))
	if err != nil {
		t.Errorf(err.Error())
	}
	if sortLines(string(data)) != sortLines(formatMap(annotations)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(annotations))
	}
	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)
}

func TestWriteWithUnixPathBadPath(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_write_with_unix_path"
		testNamespace  = "test_metadata_namespace"
		testName       = "test_metadata_name"
	)

	labels := map[string]string{
		"key1": "value1",
		"key2": "value2"}

	fake := testclient.NewSimpleFake(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      testName,
			Namespace: testNamespace,
			Labels:    labels,
		},
	})

	pluginMgr := volume.VolumePluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, fake))
	plugin, err := pluginMgr.FindPluginByName(downwardAPIPluginName)
	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			DownwardAPI: &api.DownwardAPIVolumeSource{
				Items: []api.DownwardAPIVolumeFile{
					{Path: "this//labels", FieldRef: api.ObjectFieldSelector{
						FieldPath: "metadata.labels"}},
				}}},
	}
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID, Labels: labels}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})

	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()
	err = builder.SetUp()
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}

	var data []byte
	data, err = ioutil.ReadFile(path.Join(volumePath, "this/labels"))
	if err != nil {
		t.Errorf(err.Error())
	}

	if sortLines(string(data)) != sortLines(formatMap(labels)) {
		t.Errorf("Found `%s` expected %s", data, formatMap(labels))
	}
	CleanEverything(plugin, testVolumeName, volumePath, testPodUID, t)
}
