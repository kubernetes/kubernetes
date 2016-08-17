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

package configmap

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/empty_dir"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
)

func TestMakePayload(t *testing.T) {
	cases := []struct {
		name      string
		mappings  []api.KeyToPath
		configMap *api.ConfigMap
		payload   map[string][]byte
		success   bool
	}{
		{
			name: "no overrides",
			configMap: &api.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			payload: map[string][]byte{
				"foo": []byte("foo"),
				"bar": []byte("bar"),
			},
			success: true,
		},
		{
			name: "basic 1",
			mappings: []api.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/foo.txt",
				},
			},
			configMap: &api.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			payload: map[string][]byte{
				"path/to/foo.txt": []byte("foo"),
			},
			success: true,
		},
		{
			name: "subdirs",
			mappings: []api.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
			},
			configMap: &api.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			payload: map[string][]byte{
				"path/to/1/2/3/foo.txt": []byte("foo"),
			},
			success: true,
		},
		{
			name: "subdirs 2",
			mappings: []api.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
			},
			configMap: &api.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			payload: map[string][]byte{
				"path/to/1/2/3/foo.txt": []byte("foo"),
			},
			success: true,
		},
		{
			name: "subdirs 3",
			mappings: []api.KeyToPath{
				{
					Key:  "foo",
					Path: "path/to/1/2/3/foo.txt",
				},
				{
					Key:  "bar",
					Path: "another/path/to/the/esteemed/bar.bin",
				},
			},
			configMap: &api.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			payload: map[string][]byte{
				"path/to/1/2/3/foo.txt":                []byte("foo"),
				"another/path/to/the/esteemed/bar.bin": []byte("bar"),
			},
			success: true,
		},
		{
			name: "non existent key",
			mappings: []api.KeyToPath{
				{
					Key:  "zab",
					Path: "path/to/foo.txt",
				},
			},
			configMap: &api.ConfigMap{
				Data: map[string]string{
					"foo": "foo",
					"bar": "bar",
				},
			},
			success: false,
		},
	}

	for _, tc := range cases {
		actualPayload, err := makePayload(tc.mappings, tc.configMap)
		if err != nil && tc.success {
			t.Errorf("%v: unexpected failure making payload: %v", tc.name, err)
			continue
		}

		if err == nil && !tc.success {
			t.Errorf("%v: unexpected success making payload", tc.name)
			continue
		}

		if !tc.success {
			continue
		}

		if e, a := tc.payload, actualPayload; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: expected and actual payload do not match", tc.name)
		}
	}
}

func newTestHost(t *testing.T, clientset clientset.Interface) (string, volume.VolumeHost) {
	tempDir, err := ioutil.TempDir("/tmp", "configmap_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}

	return tempDir, volumetest.NewFakeVolumeHost(tempDir, clientset, empty_dir.ProbeVolumePlugins(), "" /* rootContext */)
}

func TestCanSupport(t *testing.T) {
	pluginMgr := volume.VolumePluginMgr{}
	_, host := newTestHost(t, nil)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(configMapPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.GetPluginName() != configMapPluginName {
		t.Errorf("Wrong name: %s", plugin.GetPluginName())
	}
	if !plugin.CanSupport(&volume.Spec{Volume: &api.Volume{VolumeSource: api.VolumeSource{ConfigMap: &api.ConfigMapVolumeSource{LocalObjectReference: api.LocalObjectReference{Name: ""}}}}}) {
		t.Errorf("Expected true")
	}
	if plugin.CanSupport(&volume.Spec{}) {
		t.Errorf("Expected false")
	}
}

func TestPlugin(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_configmap_namespace"
		testName       = "test_configmap_name"

		volumeSpec = volumeSpec(testVolumeName, testName)
		configMap  = configMap(testNamespace, testName)
		client     = fake.NewSimpleClientset(&configMap)
		pluginMgr  = volume.VolumePluginMgr{}
		_, host    = newTestHost(t, client)
	)

	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(configMapPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	vName, err := plugin.GetVolumeName(volume.NewSpecFromVolume(volumeSpec))
	if err != nil {
		t.Errorf("Failed to GetVolumeName: %v", err)
	}
	if vName != "test_volume_name/test_configmap_name" {
		t.Errorf("Got unexpect VolumeName %v", vName)
	}

	volumePath := mounter.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid/volumes/kubernetes.io~configmap/test_volume_name")) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	fsGroup := int64(1001)
	err = mounter.SetUp(&fsGroup)
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	doTestConfigMapDataInVolume(volumePath, configMap, t)
	doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

// Test the case where the plugin's ready file exists, but the volume dir is not a
// mountpoint, which is the state the system will be in after reboot.  The dir
// should be mounter and the configMap data written to it.
func TestPluginReboot(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid3")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_configmap_namespace"
		testName       = "test_configmap_name"

		volumeSpec    = volumeSpec(testVolumeName, testName)
		configMap     = configMap(testNamespace, testName)
		client        = fake.NewSimpleClientset(&configMap)
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)

	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(configMapPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: testNamespace, UID: testPodUID}}
	mounter, err := plugin.NewMounter(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{})
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}

	podMetadataDir := fmt.Sprintf("%v/pods/test_pod_uid3/plugins/kubernetes.io~configmap/test_volume_name", rootDir)
	util.SetReady(podMetadataDir)
	volumePath := mounter.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid3/volumes/kubernetes.io~configmap/test_volume_name")) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	fsGroup := int64(1001)
	err = mounter.SetUp(&fsGroup)
	if err != nil {
		t.Errorf("Failed to setup volume: %v", err)
	}
	if _, err := os.Stat(volumePath); err != nil {
		if os.IsNotExist(err) {
			t.Errorf("SetUp() failed, volume path not created: %s", volumePath)
		} else {
			t.Errorf("SetUp() failed: %v", err)
		}
	}

	doTestConfigMapDataInVolume(volumePath, configMap, t)
	doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

func volumeSpec(volumeName, configMapName string) *api.Volume {
	return &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			ConfigMap: &api.ConfigMapVolumeSource{
				LocalObjectReference: api.LocalObjectReference{
					Name: configMapName,
				},
			},
		},
	}
}

func configMap(namespace, name string) api.ConfigMap {
	return api.ConfigMap{
		ObjectMeta: api.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string]string{
			"data-1": "value-1",
			"data-2": "value-2",
			"data-3": "value-3",
		},
	}
}

func doTestConfigMapDataInVolume(volumePath string, configMap api.ConfigMap, t *testing.T) {
	for key, value := range configMap.Data {
		configMapDataHostPath := path.Join(volumePath, key)
		if _, err := os.Stat(configMapDataHostPath); err != nil {
			t.Fatalf("SetUp() failed, couldn't find configMap data on disk: %v", configMapDataHostPath)
		} else {
			actualValue, err := ioutil.ReadFile(configMapDataHostPath)
			if err != nil {
				t.Fatalf("Couldn't read configMap data from: %v", configMapDataHostPath)
			}

			if value != string(actualValue) {
				t.Errorf("Unexpected value; expected %q, got %q", value, actualValue)
			}
		}
	}
}

func doTestCleanAndTeardown(plugin volume.VolumePlugin, podUID types.UID, testVolumeName, volumePath string, t *testing.T) {
	unmounter, err := plugin.NewUnmounter(testVolumeName, podUID)
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
}
