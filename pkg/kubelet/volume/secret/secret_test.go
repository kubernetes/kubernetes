/*
Copyright 2015 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

func newTestHost(t *testing.T, fakeKubeClient client.Interface) volume.Host {
	tempDir, err := ioutil.TempDir("/tmp", "secret_volume_test.")
	if err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	}

	return &volume.FakeHost{tempDir, fakeKubeClient}
}

func TestCanSupport(t *testing.T) {
	pluginMgr := volume.PluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, nil))

	plugin, err := pluginMgr.FindPluginByName(secretPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plugin.Name() != secretPluginName {
		t.Errorf("Wrong name: %s", plugin.Name())
	}
	if !plugin.CanSupport(&api.Volume{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{Target: api.ObjectReference{}}}}) {
		t.Errorf("Expected true")
	}
}

func TestPlugin(t *testing.T) {
	var (
		testPodUID     = "test_pod_uid"
		testVolumeName = "test_volume_name"
		testNamespace  = "test_secret_namespace"
		testName       = "test_secret_name"
	)

	volumeSpec := &api.Volume{
		Name: testVolumeName,
		VolumeSource: api.VolumeSource{
			Secret: &api.SecretVolumeSource{
				Target: api.ObjectReference{
					Namespace: testNamespace,
					Name:      testName,
				},
			},
		},
	}

	secret := api.Secret{
		ObjectMeta: api.ObjectMeta{
			Namespace: testNamespace,
			Name:      testName,
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1"),
			"data-2": []byte("value-2"),
			"data-3": []byte("value-3"),
		},
	}

	client := &client.Fake{
		Secret: secret,
	}

	pluginMgr := volume.PluginMgr{}
	pluginMgr.InitPlugins(ProbeVolumePlugins(), newTestHost(t, client))

	plugin, err := pluginMgr.FindPluginByName(secretPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}

	builder, err := plugin.NewBuilder(volumeSpec, &api.ObjectReference{UID: types.UID(testPodUID)})
	if err != nil {
		t.Errorf("Failed to make a new Builder: %v", err)
	}
	if builder == nil {
		t.Errorf("Got a nil Builder: %v")
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

	cleaner, err := plugin.NewCleaner(testVolumeName, types.UID(testPodUID))
	if err != nil {
		t.Errorf("Failed to make a new Cleaner: %v", err)
	}
	if cleaner == nil {
		t.Errorf("Got a nil Cleaner: %v")
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
