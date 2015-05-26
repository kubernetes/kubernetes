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

package secret

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/empty_dir"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume/util"
)

func newTestHost(t *testing.T, client client.Interface) (string, volume.VolumeHost) {
	tempDir, err := ioutil.TempDir("/tmp", "secret_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}

	return tempDir, volume.NewFakeVolumeHost(tempDir, client, empty_dir.ProbeVolumePlugins())
}

func TestCanSupport(t *testing.T) {
	pluginMgr := volume.VolumePluginMgr{}
	_, host := newTestHost(t, nil)
	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(secretPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.Name() != secretPluginName {
		t.Errorf("Wrong name: %s", plugin.Name())
	}
	if !plugin.CanSupport(&volume.Spec{Name: "foo", VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: ""}}}) {
		t.Errorf("Expected true")
	}
}

func TestPlugin(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_secret_namespace"
		testName       = "test_secret_name"

		volumeSpec = volumeSpec(testVolumeName, testName)
		secret     = secret(testNamespace, testName)
		client     = testclient.NewSimpleFake(&secret)
		pluginMgr  = volume.VolumePluginMgr{}
		_, host    = newTestHost(t, client)
	)

	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(secretPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	volumePath := builder.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid/volumes/kubernetes.io~secret/test_volume_name")) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	err = builder.SetUp()
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

	doTestSecretDataInVolume(volumePath, secret, t)
	doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

// Test the case where the 'ready' file has been created and the pod volume dir
// is a mountpoint.  Mount should not be called.
func TestPluginIdempotent(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid2")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_secret_namespace"
		testName       = "test_secret_name"

		volumeSpec    = volumeSpec(testVolumeName, testName)
		secret        = secret(testNamespace, testName)
		client        = testclient.NewSimpleFake(&secret)
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)

	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(secretPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	podVolumeDir := fmt.Sprintf("%v/pods/test_pod_uid2/volumes/kubernetes.io~secret/test_volume_name", rootDir)
	podMetadataDir := fmt.Sprintf("%v/pods/test_pod_uid2/plugins/kubernetes.io~secret/test_volume_name", rootDir)
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID}}
	mounter := &mount.FakeMounter{}
	mounter.MountPoints = []mount.MountPoint{
		{
			Path: podVolumeDir,
		},
	}
	util.SetReady(podMetadataDir)
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, mounter)
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

	if len(mounter.Log) != 0 {
		t.Errorf("Unexpected calls made to mounter: %v", mounter.Log)
	}

	if _, err := os.Stat(volumePath); err != nil {
		if !os.IsNotExist(err) {
			t.Errorf("SetUp() failed unexpectedly: %v", err)
		}
	} else {
		t.Errorf("volume path should not exist: %v", volumePath)
	}
}

// Test the case where the plugin's ready file exists, but the volume dir is not a
// mountpoint, which is the state the system will be in after reboot.  The dir
// should be mounter and the secret data written to it.
func TestPluginReboot(t *testing.T) {
	var (
		testPodUID     = types.UID("test_pod_uid3")
		testVolumeName = "test_volume_name"
		testNamespace  = "test_secret_namespace"
		testName       = "test_secret_name"

		volumeSpec    = volumeSpec(testVolumeName, testName)
		secret        = secret(testNamespace, testName)
		client        = testclient.NewSimpleFake(&secret)
		pluginMgr     = volume.VolumePluginMgr{}
		rootDir, host = newTestHost(t, client)
	)

	pluginMgr.InitPlugins(ProbeVolumePlugins(), host)

	plugin, err := pluginMgr.FindPluginByName(secretPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: testPodUID}}
	builder, err := plugin.NewBuilder(volume.NewSpecFromVolume(volumeSpec), pod, volume.VolumeOptions{}, &mount.FakeMounter{})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder")
	}

	podMetadataDir := fmt.Sprintf("%v/pods/test_pod_uid3/plugins/kubernetes.io~secret/test_volume_name", rootDir)
	util.SetReady(podMetadataDir)
	volumePath := builder.GetPath()
	if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid3/volumes/kubernetes.io~secret/test_volume_name")) {
		t.Errorf("Got unexpected path: %s", volumePath)
	}

	err = builder.SetUp()
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

	doTestSecretDataInVolume(volumePath, secret, t)
	doTestCleanAndTeardown(plugin, testPodUID, testVolumeName, volumePath, t)
}

func volumeSpec(volumeName, secretName string) *api.Volume {
	return &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			Secret: &api.SecretVolumeSource{
				SecretName: secretName,
			},
		},
	}
}

func secret(namespace, name string) api.Secret {
	return api.Secret{
		ObjectMeta: api.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1"),
			"data-2": []byte("value-2"),
			"data-3": []byte("value-3"),
		},
	}
}

func doTestSecretDataInVolume(volumePath string, secret api.Secret, t *testing.T) {
	for key, value := range secret.Data {
		secretDataHostPath := path.Join(volumePath, key)
		if _, err := os.Stat(secretDataHostPath); err != nil {
			t.Fatalf("SetUp() failed, couldn't find secret data on disk: %v", secretDataHostPath)
		} else {
			actualSecretBytes, err := ioutil.ReadFile(secretDataHostPath)
			if err != nil {
				t.Fatalf("Couldn't read secret data from: %v", secretDataHostPath)
			}

			actualSecretValue := string(actualSecretBytes)
			if string(value) != actualSecretValue {
				t.Errorf("Unexpected value; expected %q, got %q", value, actualSecretValue)
			}
		}
	}
}

func doTestCleanAndTeardown(plugin volume.VolumePlugin, podUID types.UID, testVolumeName, volumePath string, t *testing.T) {
	cleaner, err := plugin.NewCleaner(testVolumeName, podUID, mount.New())
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
