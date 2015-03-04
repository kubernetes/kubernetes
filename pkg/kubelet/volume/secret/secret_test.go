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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	if !plugin.CanSupport(&api.Volume{Source: api.VolumeSource{Secret: &api.SecretVolumeSource{Target: api.ObjectReference{}}}}) {
		t.Errorf("Expected true")
	}
}

func TestConvertKeyToVar(t *testing.T) {
	cases := []struct {
		input    string
		expected string
	}{
		{"id-rsa.pub", "ID_RSA_PUB"},
		{"a-long-string-with-dashes", "A_LONG_STRING_WITH_DASHES"},
		{"lots.of.dots", "LOTS_OF_DOTS"},
	}

	for _, tc := range cases {
		actual := convertKeyToVar(tc.input)
		if tc.expected != actual {
			t.Errorf("Unexpected converted value; got '%v', expected '%v'", actual, tc.expected)
		}
	}
}

func TestMakeEnvFileContent(t *testing.T) {
	secret := okSecret()

	cases := []struct {
		name     string
		secret   api.Secret
		env      api.SecretEnv
		expected string
	}{
		{
			name:   "no adaptations",
			secret: secret,
			env:    api.SecretEnv{Adaptations: []api.StringAdaptation{}},
			expected: strings.TrimLeft(`
export DATA_1="value-1"
export DATA_2="value-2"
export DATA_3="value-3"
`, "\n"),
		},
		{
			name:   "all adaptations",
			secret: secret,
			env: api.SecretEnv{
				Adaptations: []api.StringAdaptation{
					{From: "data-1", To: "ADAPTED_VALUE_1"},
					{From: "data-2", To: "ADAPTED_VALUE_2"},
					{From: "data-3", To: "ADAPTED_VALUE_3"},
				},
			},
			expected: strings.TrimLeft(`
export ADAPTED_VALUE_1="value-1"
export ADAPTED_VALUE_2="value-2"
export ADAPTED_VALUE_3="value-3"
`, "\n")},
		{
			name:   "mixed",
			secret: secret,
			env: api.SecretEnv{
				Adaptations: []api.StringAdaptation{
					{From: "data-1", To: "ADAPTED_VALUE_1"},
					{From: "data-2", To: "ADAPTED_VALUE_2"},
				},
			},
			expected: strings.TrimLeft(`
export ADAPTED_VALUE_1="value-1"
export ADAPTED_VALUE_2="value-2"
export DATA_3="value-3"
`, "\n")},
	}

	for _, tc := range cases {
		actual := makeEnvFileContent(&tc.secret, &tc.env)
		matchStringsByLine(t, tc.name, tc.expected, actual)
	}
}

func TestPlugin(t *testing.T) {
	var (
		testPodUID    = "test_pod_uid"
		testNamespace = "test_secret_namespace"
		testName      = "test_secret_name"
	)

	secret := okSecret()

	cases := []struct {
		name     string
		volume   *api.Volume
		verifier func(t *testing.T, tcName, volumePath string, secret api.Secret)
	}{
		{
			name: "discrete files",
			volume: &api.Volume{
				Name: "files-volume",
				Source: api.VolumeSource{
					Secret: &api.SecretVolumeSource{
						Target: api.ObjectReference{
							Namespace: testNamespace,
							Name:      testName,
						},
					},
				},
			},
			verifier: checkSecretDataInDiscreteFiles,
		},
		{
			name: "env file",
			volume: &api.Volume{
				Name: "files-volume",
				Source: api.VolumeSource{
					Secret: &api.SecretVolumeSource{
						Target: api.ObjectReference{
							Namespace: testNamespace,
							Name:      testName,
						},
						EnvAdaptations: &api.SecretEnv{
							Name: "environment-file",
							Adaptations: []api.StringAdaptation{
								{From: "data-1", To: "ADAPTED_VALUE_1"},
								{From: "data-2", To: "ADAPTED_VALUE_2"},
								{From: "data-3", To: "ADAPTED_VALUE_3"},
							},
						},
					},
				},
			},

			verifier: newEnvFileChecker("environment-file", strings.TrimLeft(`
export ADAPTED_VALUE_1="value-1"
export ADAPTED_VALUE_2="value-2"
export ADAPTED_VALUE_3="value-3"
`, "\n")).verify,
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

	for _, tc := range cases {
		builder, err := plugin.NewBuilder(tc.volume, types.UID(testPodUID))
		if err != nil {
			t.Errorf("Failed to make a new Builder: %v", err)
		}
		if builder == nil {
			t.Errorf("Got a nil Builder: %v")
		}

		volumePath := builder.GetPath()
		if !strings.HasSuffix(volumePath, fmt.Sprintf("pods/test_pod_uid/volumes/kubernetes.io~secret/%v", tc.volume.Name)) {
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

		tc.verifier(t, tc.name, volumePath, secret)

		cleaner, err := plugin.NewCleaner(tc.volume.Name, types.UID(testPodUID))
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
}

func okSecret() api.Secret {
	return api.Secret{
		ObjectMeta: api.ObjectMeta{
			Namespace: "test-namespace",
			Name:      "test-secret-name",
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1"),
			"data-2": []byte("value-2"),
			"data-3": []byte("value-3"),
		},
	}
}

func matchStringsByLine(t *testing.T, tcName, expected, actual string) {
	// Really ridiculous way of comparing strings, but necessary because
	// ordering is not guaranteed while traversing a map.
	expectedLines := util.NewStringSet(strings.Split(expected, "\n")...)
	actualLines := util.NewStringSet(strings.Split(actual, "\n")...)

	for _, expectedLine := range expectedLines.List() {
		if actualLines.Has(expectedLine) {
			actualLines.Delete(expectedLine)
		} else {
			t.Errorf("%v: Actual output missing expected line: '%v'", tcName, expectedLine)
		}

		expectedLines.Delete(expectedLine)
	}

	if actualLines.Len() > 0 {
		t.Errorf("%v: Unexpected actual lines of output: %v", tcName, actualLines)
	}
}

func checkSecretDataInDiscreteFiles(t *testing.T, tcName, volumePath string, secret api.Secret) {
	for key, value := range secret.Data {
		secretDataHostPath := path.Join(volumePath, key)
		if _, err := os.Stat(secretDataHostPath); err != nil {
			t.Fatalf("%v: SetUp() failed, couldn't find secret data on disk: %v", tcName, secretDataHostPath)
		} else {
			actualSecretBytes, err := ioutil.ReadFile(secretDataHostPath)
			if err != nil {
				t.Fatalf("%v: Couldn't read secret data from: %v", tcName, secretDataHostPath)
			}

			actualSecretValue := string(actualSecretBytes)
			if string(value) != actualSecretValue {
				t.Errorf("%v: Unexpected value; expected %q, got %q", tcName, value, actualSecretValue)
			}
		}
	}
}

type envFileChecker struct {
	name     string
	expected string
}

func newEnvFileChecker(name, expected string) *envFileChecker {
	return &envFileChecker{name, expected}
}

func (c *envFileChecker) verify(t *testing.T, tcName, volumePath string, secret api.Secret) {
	envFileHostPath := path.Join(volumePath, c.name)
	if _, err := os.Stat(envFileHostPath); err != nil {
		t.Fatalf("%v: SetUp() failed, couldn't find secret data on disk: %v", tcName, envFileHostPath)
	} else {
		actualBytes, err := ioutil.ReadFile(envFileHostPath)
		if err != nil {
			t.Fatalf("%v: Couldn't read env file from: %v", tcName, envFileHostPath)
		}

		actual := string(actualBytes)
		matchStringsByLine(t, tcName, c.expected, actual)
	}
}
